"""Assay ledger puller v3.1: reads ASSAY-Bot's Metaculus forecasts, community
predictions, and resolutions into assay/ledger.json. Runs in GitHub Actions.
Token never leaves GitHub secrets; output is public data.

Design (2026-10-06), aligned with the maintained Metaculus forecasting-tools
client (metaculus_client.py, Oct 2026):
- Offset pagination (limit=100, offset=N) exactly like the reference client.
  v2 followed "next" URLs and could loop forever on a silent success path;
  offset paging with a zero-new-IDs stop cannot.
- Listing carries full question JSON incl. community aggregates
  (with_cp=true). Per-post GETs only as a fallback when a listing item
  lacks my_forecasts.
- Never silent: every page, retry, and fallback prints with flush=True.
- Hard time budget: on exhaustion, writes a PARTIAL ledger and exits 0.
  The workflow's timeout-minutes is the outer backstop.
- Listing failures after retries still fail the run (dead-man semantics:
  a stale ledger is the visible signal). Per-post fallback failures skip.
- Ledger write is atomic (tmp file + os.replace).
- v3.1: p extracted from forecast_values=[p_no,p_yes] (current API shape;
  legacy probability_yes kept as fallback); forecasters_count fallback for
  n_forecasters; community_raw blob stored for shape discovery.
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import requests

TOKEN = os.environ["METACULUS_TOKEN"]
BASE = "https://www.metaculus.com/api"
TOURNAMENTS = [t.strip() for t in
               os.environ.get("ASSAY_TOURNAMENTS", "33121,minibench").split(",")
               if t.strip()]
LEDGER = "assay/ledger.json"
STATUSES = ["open", "closed", "resolved"]

PAGE_SIZE = 100
MAX_PAGES = 25          # hard stop, loop-proof: 25*100 = 2500 posts max
PACE = 1.2              # seconds between successful calls (+ jitter)
MAX_TRIES = 4           # per request before giving up
MAX_BACKOFF = 60        # seconds; Retry-After is honored but capped here
BUDGET_S = 1200         # 20 min; workflow timeout-minutes: 30 is the backstop
MAX_FALLBACKS = 50      # per-post GETs allowed when listing lacks my_forecasts

SESSION = requests.Session()
SESSION.headers["Authorization"] = "Token " + TOKEN
SESSION.headers["User-Agent"] = "assay-ledger/3.1 (github-actions)"

T0 = time.monotonic()
PARTIAL = False


class PostFetchError(Exception):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def elapsed():
    return time.monotonic() - T0


def over_budget():
    return elapsed() > BUDGET_S


def log(msg):
    print("[%5.0fs] %s" % (elapsed(), msg), flush=True)


def _retry_after(resp, default):
    try:
        return min(float(resp.headers.get("Retry-After") or default), MAX_BACKOFF)
    except (TypeError, ValueError):
        return default


def get(url, params=None, tries=MAX_TRIES):
    """GET JSON with pacing and retry. Raises PostFetchError if exhausted."""
    delay = 5.0
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            last = e
            log("NET %s on %s, retry %d/%d in %.0fs"
                % (type(e).__name__, url[-60:], attempt, tries, delay))
            time.sleep(delay + random.uniform(0, delay * 0.5))
            delay = min(delay * 2, MAX_BACKOFF)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = _retry_after(r, delay) + random.uniform(0, 2)
            log("HTTP %d on %s, retry %d/%d in %.0fs"
                % (r.status_code, url[-60:], attempt, tries, wait))
            time.sleep(wait)
            delay = min(delay * 2, MAX_BACKOFF)
            continue
        if r.status_code >= 400:
            raise PostFetchError("HTTP %d on %s (not retryable)"
                                 % (r.status_code, url[-60:]))
        time.sleep(PACE + random.uniform(0, 0.8))
        return r.json()
    raise PostFetchError("giving up after %d tries on %s (last: %s)"
                         % (tries, url[-60:], last))


def norm_y(resolution, qtype):
    if resolution is None:
        return None
    if qtype == "binary":
        r = str(resolution).lower()
        return 1.0 if r == "yes" else (0.0 if r == "no" else None)
    try:
        return float(resolution)
    except (TypeError, ValueError):
        return None


def my_p(latest, qtype):
    """My probability_yes from a forecast object. Current API: binary forecasts
    carry forecast_values=[p_no, p_yes]; legacy shape used probability_yes."""
    if not latest or qtype != "binary":
        return None
    fv = latest.get("forecast_values")
    if fv and len(fv) == 2:
        return float(fv[1])
    if latest.get("probability_yes") is not None:
        return float(latest["probability_yes"])
    return None


def community_p(q):
    """Community center for binary questions; try every known API shape."""
    try:
        agg = q.get("aggregations") or {}
        for method in ("recency_weighted", "unweighted"):
            latest = (agg.get(method) or {}).get("latest") or {}
            centers = latest.get("centers")
            if centers:
                return float(centers[0])
        for key in ("community_prediction", "metaculus_prediction"):
            full = (q.get(key) or {}).get("full") or {}
            if full.get("q2") is not None:
                return float(full["q2"])
    except Exception:
        pass
    return None


def community_raw(q):
    """Compact copy of the community aggregation blob, kept for shape
    discovery — the API has renamed fields on us before."""
    agg = q.get("aggregations") or {}
    out = {}
    for method in ("recency_weighted", "unweighted"):
        latest = (agg.get(method) or {}).get("latest")
        if latest:
            out[method] = latest
    return out or None


def question_record(q, post, tournament):
    latest = (q.get("my_forecasts") or {}).get("latest") or {}
    ftime = latest.get("start_time")
    is_binary = q.get("type") == "binary"
    return {
        "question_id": q.get("id"),
        "post_id": post.get("id"),
        "tournament": tournament,
        "title": q.get("title") or post.get("title"),
        "type": q.get("type"),
        "status": q.get("status"),
        "open_time": q.get("open_time"),
        "scheduled_close_time": q.get("scheduled_close_time"),
        "scheduled_resolve_time": q.get("scheduled_resolve_time"),
        "producer": "gemini-2.5-flash via openrouter (metaculus template)",
        "p": my_p(latest, q.get("type")),
        "forecast_time": (
            datetime.fromtimestamp(ftime, timezone.utc).isoformat() if ftime else None
        ),
        "forecast_raw": latest or None,
        "p_community": community_p(q) if is_binary else None,
        "community_raw": community_raw(q),
        "n_forecasters": q.get("nr_forecasters") or q.get("forecasters_count"),
        "resolution": q.get("resolution"),
        "y": norm_y(q.get("resolution"), q.get("type")),
        "actual_resolve_time": q.get("actual_resolve_time"),
    }


def iter_post_stubs(tid):
    """Yield post stubs via offset pagination (reference-client pattern).
    Stops on: short page, zero new post IDs, page cap, or time budget."""
    seen_post_ids = set()
    for page_num in range(MAX_PAGES):
        if over_budget():
            log("BUDGET exhausted during listing of %s; stopping" % tid)
            break
        params = {
            "limit": PAGE_SIZE,
            "offset": page_num * PAGE_SIZE,
            "tournaments": [tid],
            "statuses": STATUSES,
            "with_cp": "true",
        }
        page = get(BASE + "/posts/", params=params)
        results = page.get("results", [])
        fresh = [st for st in results if st.get("id") not in seen_post_ids]
        log("tournament %s page %d: %d posts (%d new)"
            % (tid, page_num, len(results), len(fresh)))
        for stub in fresh:
            seen_post_ids.add(stub.get("id"))
            if "question" in stub or "group_of_questions" in stub:
                yield stub
        if len(results) < PAGE_SIZE or not fresh:
            break


def main():
    global PARTIAL
    records = {}
    skipped = []
    fallbacks = 0
    for tid in TOURNAMENTS:
        n_seen = 0
        for stub in iter_post_stubs(tid):
            n_seen += 1
            questions = []
            if "question" in stub:
                questions = [stub["question"]]
            else:
                questions = stub["group_of_questions"].get("questions", [])
            needs_detail = any("my_forecasts" not in q for q in questions)
            if needs_detail and fallbacks < MAX_FALLBACKS and not over_budget():
                try:
                    fallbacks += 1
                    post = get(BASE + "/posts/%s/" % stub["id"])
                    questions = ([post["question"]] if "question" in post
                                 else post["group_of_questions"].get("questions", []))
                except PostFetchError as e:
                    skipped.append(stub.get("id"))
                    log("SKIP post %s: %s" % (stub.get("id"), e))
                    continue
            elif needs_detail:
                log("no fallback left for post %s; using listing data"
                    % stub.get("id"))
            for q in questions:
                rec = question_record(q, stub, tid)
                records[str(rec["question_id"])] = rec
        log("tournament %s: %d posts scanned" % (tid, n_seen))
        if over_budget():
            PARTIAL = True
            log("BUDGET exhausted; writing partial ledger")
            break

    old = {}
    if os.path.exists(LEDGER):
        old = json.load(open(LEDGER))
    for qid, rec in records.items():
        rec["first_seen_at"] = old.get(qid, {}).get("first_seen_at", now_iso())
        rec["last_updated_at"] = now_iso()
        old[qid] = rec

    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER + ".tmp", "w") as f:
        json.dump(old, f, indent=2, sort_keys=True)
    os.replace(LEDGER + ".tmp", LEDGER)

    npred = sum(1 for r in old.values() if r["p"] is not None)
    nres = sum(1 for r in old.values() if r["y"] is not None)
    ncp = sum(1 for r in old.values() if r["p_community"] is not None)
    log("ledger%s: %d questions | with forecast: %d | with community p: %d | resolved: %d | skipped posts: %d | fallbacks: %d"
        % (" (PARTIAL)" if PARTIAL else "", len(old), npred, ncp, nres,
           len(skipped), fallbacks))
    for r in sorted(old.values(), key=lambda x: x["question_id"] or 0):
        log("q%s | %s | p=%s | cp=%s | y=%s | %s" % (
            r["question_id"], r["status"], r["p"], r["p_community"], r["y"],
            (r["title"] or "")[:45]))


if __name__ == "__main__":
    main()
