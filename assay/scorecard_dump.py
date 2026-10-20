#!/usr/bin/env python3
"""assay/scorecard_dump.py — SCORECARD-DUMP v2
Read-only Metaculus question dump feeding the box-side scorecard.

Runs ONLY inside GitHub Actions: METACULUS_TOKEN lives in Actions secrets and
never leaves them. Reads assay/ensemble_members.jsonl from the checkout,
fetches each unique question's public data via the authenticated API, trims
it, writes assay/scorecard_dump.json (committed by the workflow).

v2: cache-merge (loads the existing dump, keeps good records, retries only
missing/errored ones; REFETCH_ALL=1 forces a full refetch), proper 429
backoff honoring Retry-After, slower pace, richer schema eyeball.

Stdlib only. Fail-open everywhere: per-question errors are recorded in the
dump, never raised. Prints SCORECARD-DUMP marker lines for log greps.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

MARKER = "SCORECARD-DUMP"
API_BASE = "https://www.metaculus.com/api"
HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "ensemble_members.jsonl")
OUT = os.path.join(HERE, "scorecard_dump.json")
PAGE_QID = re.compile(r"/questions/(\d+)")
PACE = 1.2          # between questions
BACKOFF_429 = 12.0  # base wait on rate limit (Retry-After wins if larger)
BACKOFF_5XX = 5.0
MAX_TRIES = 4


def now_z():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ids():
    """Unique question ids from the ledger, first-seen order.
    Falls back to the /questions/<id> page_url regex when question_id is absent."""
    ids, order = set(), []
    if not os.path.exists(JSONL):
        print(f"{MARKER}: no ledger at {JSONL} — writing empty dump")
        return order
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qid = rec.get("question_id")
            if qid is None:
                m = PAGE_QID.search(rec.get("page_url") or "")
                qid = m.group(1) if m else None
            try:
                qid = int(qid)
            except (TypeError, ValueError):
                continue
            if qid not in ids:
                ids.add(qid)
                order.append(qid)
    return order


def load_cache():
    if not os.path.exists(OUT):
        return {}
    try:
        with open(OUT) as f:
            d = json.load(f)
        q = d.get("questions")
        return q if isinstance(q, dict) else {}
    except Exception:
        return {}


def fetch_once(qid, token):
    """Single attempt: (status, question-dict|None, retry_after|None)."""
    url = f"{API_BASE}/questions/{qid}/"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Token {token}",
        "User-Agent": "assay-scorecard-dump",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        return e.code, None, e.headers.get("Retry-After")
    except Exception as e:
        return f"err_{type(e).__name__}", None, None


def fetch(qid, token):
    """With retries/backoff. Returns (status, question-dict|None)."""
    st, q, ra = None, None, None
    for attempt in range(1, MAX_TRIES + 1):
        st, q, ra = fetch_once(qid, token)
        if st == 200:
            return st, q
        if st in (401, 403):
            return st, None  # auth problem: retrying won't help
        if st == 429:
            wait = BACKOFF_429
            try:
                if ra is not None:
                    wait = max(wait, float(ra))
            except (TypeError, ValueError):
                pass
            print(f"{MARKER}: q#{qid} 429 — backoff {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_TRIES})")
            time.sleep(wait)
            continue
        if isinstance(st, int) and st >= 500:
            print(f"{MARKER}: q#{qid} {st} — retry in {BACKOFF_5XX:.0f}s "
                  f"(attempt {attempt}/{MAX_TRIES})")
            time.sleep(BACKOFF_5XX)
            continue
        return st, None  # 404 etc: permanent, don't retry
    return st, None


def _first_num(container):
    for key in ("means", "centers"):
        v = container.get(key)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return float(v[0])
    v = container.get("center")
    return float(v) if isinstance(v, (int, float)) else None


def trim(qid, q):
    """Keep only what the scorecard needs. Defensive .get throughout —
    schema drift downgrades a record, it never crashes the dump."""
    agg = q.get("aggregations") or {}
    rw = agg.get("recency_weighted") or {}
    hist = []
    for h in rw.get("history") or []:
        if not isinstance(h, dict):
            continue
        hist.append({
            "t0": h.get("start_time"),
            "t1": h.get("end_time"),
            "p": _first_num(h),
        })
    latest = rw.get("latest") or {}
    return {
        "question_id": qid,
        "title": q.get("title"),
        "type": q.get("type"),
        "status": q.get("status"),
        "resolution": q.get("resolution"),
        "scheduled_close_time": q.get("scheduled_close_time"),
        "actual_close_time": q.get("actual_close_time"),
        "n_history": len(hist),
        "history": hist,
        "latest_p": _first_num(latest) if isinstance(latest, dict) else None,
        "fetched_ts": now_z(),
    }


def main():
    token = os.environ.get("METACULUS_TOKEN", "").strip()
    if not token:
        print(f"{MARKER}: METACULUS_TOKEN missing from env — aborting, no dump written")
        return 1
    refetch_all = os.environ.get("REFETCH_ALL", "").strip().lower() in ("1", "true", "yes")
    ids = load_ids()
    cache = {} if refetch_all else load_cache()
    out = dict(cache)
    todo = [qid for qid in ids
            if refetch_all or str(qid) not in cache or "error" in (cache.get(str(qid)) or {})]
    print(f"{MARKER}: {len(ids)} unique ids in ledger — cached_ok="
          f"{len(ids) - len(todo)} to_fetch={len(todo)} refetch_all={refetch_all}")

    n_ok, n_fail = 0, 0
    eyeballed_keys = eyeballed_resolved = False
    for i, qid in enumerate(todo):
        st, q = fetch(qid, token)
        if st == 200 and isinstance(q, dict):
            rec = trim(qid, q)
            out[str(qid)] = rec
            n_ok += 1
            if not eyeballed_keys:
                eyeballed_keys = True
                keys = sorted(q.keys())
                print(f"{MARKER}: raw top-level keys (q#{qid}): {keys}")
                agg = q.get("aggregations")
                if isinstance(agg, dict):
                    print(f"{MARKER}: aggregations subkeys: {sorted(agg.keys())}")
                    rw = agg.get("recency_weighted") or {}
                    print(f"{MARKER}: recency_weighted subkeys: {sorted(rw.keys())}")
                else:
                    print(f"{MARKER}: 'aggregations' key present: {agg is not None}")
            if (not eyeballed_resolved
                    and rec["resolution"] not in (None, "", "ambiguous", "annulled")):
                eyeballed_resolved = True
                print(f"{MARKER}: resolved eyeball (raw q#{qid}, first 2000 chars):")
                print(json.dumps(q)[:2000])
        else:
            out[str(qid)] = {"question_id": qid, "error": f"http_{st}",
                             "fetched_ts": now_z()}
            n_fail += 1
            print(f"{MARKER}: q#{qid} fetch failed after retries: http_{st}")
            if st in (401, 403):
                print(f"{MARKER}: auth refused on q#{qid} — aborting early; "
                      f"check token secret")
                break
        time.sleep(PACE)
        if (i + 1) % 20 == 0:
            print(f"{MARKER}: progress {i + 1}/{len(todo)} ok={n_ok} fail={n_fail}")

    n_good = sum(1 for v in out.values() if isinstance(v, dict) and "error" not in v)
    n_resolved = sum(1 for v in out.values()
                     if isinstance(v, dict) and "error" not in v
                     and v.get("resolution") not in (None, "", "ambiguous", "annulled"))
    payload = {
        "generated_ts": now_z(),
        "n_ids": len(ids),
        "n_ok": n_good,
        "n_fail": len(ids) - n_good,
        "n_resolved": n_resolved,
        "questions": out,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"{MARKER}: wrote {os.path.relpath(OUT)} ok_total={n_good} "
          f"resolved_total={n_resolved} fail_total={len(ids) - n_good} "
          f"bytes={os.path.getsize(OUT)}")
    return 0 if n_good > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
