#!/usr/bin/env python3
"""assay/scorecard_dump.py — SCORECARD-DUMP
Read-only Metaculus question dump feeding the box-side scorecard.

Runs ONLY inside GitHub Actions: METACULUS_TOKEN lives in Actions secrets and
never leaves them. Reads assay/ensemble_members.jsonl from the checkout,
fetches each unique question's public data via the authenticated API, trims
it, writes assay/scorecard_dump.json (committed by the workflow).

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
SLEEP_OK = 0.4
SLEEP_ERR = 2.0


def now_z():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ids():
    """Unique question ids from the ledger, first-seen order.
    Falls back to the /questions/<id> page_url regex when question_id is absent
    (same fallback as PR-10b)."""
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


def fetch(qid, token):
    """(status, question-dict|None). status is int HTTP code or 'err_*' string."""
    url = f"{API_BASE}/questions/{qid}/"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Token {token}",
        "User-Agent": "assay-scorecard-dump",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return f"err_{type(e).__name__}", None


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
    ids = load_ids()
    print(f"{MARKER}: {len(ids)} unique question ids in ledger")
    out, n_ok, n_fail, n_resolved = {}, 0, 0, 0
    eyeballed = False
    for i, qid in enumerate(ids):
        st, q = fetch(qid, token)
        if st == 200 and isinstance(q, dict):
            rec = trim(qid, q)
            out[str(qid)] = rec
            n_ok += 1
            if rec["resolution"] not in (None, "", "ambiguous", "annulled"):
                n_resolved += 1
            if not eyeballed:
                eyeballed = True
                print(f"{MARKER}: schema eyeball (raw q#{qid}, first 1200 chars):")
                print(json.dumps(q)[:1200])
        else:
            out[str(qid)] = {"question_id": qid, "error": f"http_{st}",
                             "fetched_ts": now_z()}
            n_fail += 1
            print(f"{MARKER}: q#{qid} fetch failed: http_{st}")
            if st in (401, 403):
                print(f"{MARKER}: auth refused on q#{qid} — aborting early; "
                      f"check token secret")
                break
            if isinstance(st, int) and st >= 500:
                time.sleep(SLEEP_ERR)
        time.sleep(SLEEP_OK)
        if (i + 1) % 25 == 0:
            print(f"{MARKER}: progress {i + 1}/{len(ids)} ok={n_ok} fail={n_fail}")
    payload = {
        "generated_ts": now_z(),
        "n_ids": len(ids),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_resolved": n_resolved,
        "questions": out,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"{MARKER}: wrote {os.path.relpath(OUT)} "
          f"ok={n_ok} resolved={n_resolved} fail={n_fail} "
          f"bytes={os.path.getsize(OUT)}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
