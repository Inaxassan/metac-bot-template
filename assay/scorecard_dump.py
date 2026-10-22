#!/usr/bin/env python3
"""assay/scorecard_dump.py — SCORECARD-DUMP v3
Read-only Metaculus question + track-record dump feeding the box-side scorecard.

Runs ONLY inside GitHub Actions: METACULUS_TOKEN lives in Actions secrets and
never leaves them. Three lanes, all read-only:

  1. question lane: every unique question id in assay/ensemble_members.jsonl
     (cache-merged; retries only missing/errored; 429 backoff).
  2. track-record lane: /api/users/me/ -> /api/users/<id>/?include_stats=true
     -> the bot's OWN official peer scores per question (score_scatter_plot),
     histogram, calibration curve, forecasting stats. Refetched each run.
  3. scored-question lane: for every post_id in the scatter plot, fetch the
     post, extract its question ids (single/group), and pull full question
     data for any not already cached — so the scored set (Cup etc., which the
     ledger doesn't track) also has resolutions + community history.

Writes assay/scorecard_dump.json (committed by the workflow). Stdlib only.
Fail-open everywhere. Prints SCORECARD-DUMP / TRACK-RECORD marker lines.
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
TR_MARKER = "TRACK-RECORD"
API_BASE = "https://www.metaculus.com/api"
HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "ensemble_members.jsonl")
OUT = os.path.join(HERE, "scorecard_dump.json")
PAGE_QID = re.compile(r"/questions/(\d+)")
PACE = 1.2
BACKOFF_429 = 12.0
BACKOFF_5XX = 5.0
MAX_TRIES = 4


def now_z():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- transport ----------

def fetch_once(url, token):
    """Single attempt: (status, payload|None, retry_after|None)."""
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


def fetch(url, token, label=""):
    """With retries/backoff. Returns (status, payload|None)."""
    st, q, ra = None, None, None
    for attempt in range(1, MAX_TRIES + 1):
        st, q, ra = fetch_once(url, token)
        if st == 200:
            return st, q
        if st in (401, 403):
            return st, None
        if st == 429:
            wait = BACKOFF_429
            try:
                if ra is not None:
                    wait = max(wait, float(ra))
            except (TypeError, ValueError):
                pass
            print(f"{MARKER}: {label} 429 — backoff {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_TRIES})")
            time.sleep(wait)
            continue
        if isinstance(st, int) and st >= 500:
            print(f"{MARKER}: {label} {st} — retry in {BACKOFF_5XX:.0f}s "
                  f"(attempt {attempt}/{MAX_TRIES})")
            time.sleep(BACKOFF_5XX)
            continue
        return st, None
    return st, None


# ---------- question lane ----------

def load_ids():
    ids, order = set(), []
    if not os.path.exists(JSONL):
        print(f"{MARKER}: no ledger at {JSONL} — question lane empty")
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


def _first_num(container):
    for key in ("means", "centers"):
        v = container.get(key)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return float(v[0])
    v = container.get("center")
    return float(v) if isinstance(v, (int, float)) else None


def trim(qid, q):
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
        "post_id": q.get("post_id"),
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


def fetch_question(qid, token, out):
    st, q = fetch(f"{API_BASE}/questions/{qid}/", token, label=f"q#{qid}")
    if st == 200 and isinstance(q, dict):
        out[str(qid)] = trim(qid, q)
        return True
    out[str(qid)] = {"question_id": qid, "error": f"http_{st}",
                     "fetched_ts": now_z()}
    print(f"{MARKER}: q#{qid} fetch failed after retries: http_{st}")
    return st not in (401, 403)


# ---------- track-record lane ----------

def qids_from_post(post):
    """Question ids from a post payload: single question or group."""
    ids = []
    q = post.get("question")
    if isinstance(q, dict) and q.get("id") is not None:
        ids.append(q["id"])
    g = post.get("group_of_questions")
    if isinstance(g, dict):
        for qq in g.get("questions") or []:
            if isinstance(qq, dict) and qq.get("id") is not None:
                ids.append(qq["id"])
    return ids


def track_record_lane(token, out):
    """Fills track_record block; adds scored questions to `out` in place."""
    st, me = fetch(f"{API_BASE}/users/me/", token, label="users/me")
    if st != 200 or not isinstance(me, dict):
        print(f"{TR_MARKER}: users/me failed: http_{st} — skipping lane")
        return None
    uid = me.get("id")
    uname = me.get("username")
    print(f"{TR_MARKER}: authed as user {uname!r} id={uid}")
    time.sleep(PACE)

    st, prof = fetch(f"{API_BASE}/users/{uid}/?include_stats=true", token,
                     label="profile stats")
    if st != 200 or not isinstance(prof, dict):
        print(f"{TR_MARKER}: profile stats failed: http_{st} — skipping lane")
        return None
    stats_keys = sorted(prof.keys())
    print(f"{TR_MARKER}: profile payload keys: {stats_keys}")
    scatter = prof.get("score_scatter_plot") or []
    print(f"{TR_MARKER}: score_scatter_plot rows: {len(scatter)}")
    if scatter:
        r0 = scatter[0]
        print(f"{TR_MARKER}: scatter row sample: {json.dumps(r0)[:300]}")
        vals = [r.get("score") for r in scatter if isinstance(r.get("score"), (int, float))]
        if vals:
            print(f"{TR_MARKER}: official avg peer score over {len(vals)} scored "
                  f"= {sum(vals) / len(vals):+.2f}")

    # scored-question lane: resolve post_ids -> question ids -> full data
    post_ids = []
    for r in scatter:
        pid = r.get("post_id")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid not in post_ids:
            post_ids.append(pid)
    n_new = 0
    for pid in post_ids:
        st, post = fetch(f"{API_BASE}/posts/{pid}/", token, label=f"post#{pid}")
        time.sleep(PACE)
        if st != 200 or not isinstance(post, dict):
            print(f"{TR_MARKER}: post#{pid} fetch failed: http_{st}")
            continue
        for qid in qids_from_post(post):
            have = out.get(str(qid))
            if isinstance(have, dict) and "error" not in have:
                continue  # already cached by the question lane
            st2 = fetch_question(qid, token, out)
            n_new += 1
            time.sleep(PACE)
            if st2 is not True:
                break
    print(f"{TR_MARKER}: {len(post_ids)} scored posts, {n_new} new questions fetched")
    return {
        "user_id": uid,
        "username": uname,
        "fetched_ts": now_z(),
        "stats": prof,
    }


# ---------- main ----------

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
    for i, qid in enumerate(todo):
        ok = fetch_question(qid, token, out)
        if ok is True:
            n_ok += 1
        elif ok is False:
            n_fail += 1
        else:  # auth refused — fetch_question already logged; abort lane
            break
        time.sleep(PACE)
        if (i + 1) % 20 == 0:
            print(f"{MARKER}: progress {i + 1}/{len(todo)} ok={n_ok} fail={n_fail}")

    track = track_record_lane(token, out)

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
        "track_record": track,
        "questions": out,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"{MARKER}: wrote {os.path.relpath(OUT)} ok_total={n_good} "
          f"resolved_total={n_resolved} fail_total={len(ids) - n_good} "
          f"track_record={'yes' if track else 'NO'} bytes={os.path.getsize(OUT)}")
    return 0 if n_good > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
