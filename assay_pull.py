"""Assay ledger puller: reads ASSAY-Bot's Metaculus forecasts, community
predictions, and resolutions into assay/ledger.json. Runs in GitHub Actions.
Token never leaves GitHub secrets; output is public data.

Design (2026-10-06):
- One requests.Session, paced calls (2.5s + jitter), exponential backoff with
  Retry-After honored. Transient 429/5xx/network errors never kill a run.
- Listing pagination followed via "next" (no silent >100-post truncation).
- A post that hard-fails after all retries is skipped with a warning; a dead
  listing endpoint still fails the run (dead-man semantics preserved).
- Ledger write is atomic (tmp file + os.replace).
- Tournaments overridable via ASSAY_TOURNAMENTS env (comma-separated).
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import requests

TOKEN = os.environ["METACULUS_TOKEN"]
BASE = "https://www.metaculus.com/api"
TOURNAMENTS = [t.strip() for t in
               os.environ.get("ASSAY_TOURNAMENTS", "33121,minibench").split(",")
               if t.strip()]
LEDGER = "assay/ledger.json"
PACE = 2.5       # seconds between successful calls
MAX_TRIES = 6    # per request before giving up

SESSION = requests.Session()
SESSION.headers["Authorization"] = "Token " + TOKEN
SESSION.headers["User-Agent"] = "assay-ledger/2.0 (github-actions)"


class PostFetchError(Exception):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _tag(url):
    parts = url.rstrip("/").rsplit("/", 2)
    return "/".join(parts[-2:])


def _retry_after(resp, default):
    try:
        return float(resp.headers.get("Retry-After") or default)
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
            print("  NET %s on %s, retry %d/%d in %.0fs"
                  % (type(e).__name__, _tag(url), attempt, tries, delay))
            time.sleep(delay + random.uniform(0, delay * 0.5))
            delay = min(delay * 2, 120)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = _retry_after(r, delay) + random.uniform(0, 2)
            print("  HTTP %d on %s, retry %d/%d in %.0fs"
                  % (r.status_code, _tag(url), attempt, tries, wait))
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        if r.status_code >= 400:
            raise PostFetchError("HTTP %d on %s (not retryable)"
                                 % (r.status_code, _tag(url)))
        time.sleep(PACE + random.uniform(0, 1.0))
        return r.json()
    raise PostFetchError("giving up after %d tries on %s (last: %s)"
                         % (tries, _tag(url), last))


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
        "p": latest.get("probability_yes") if is_binary else None,
        "forecast_time": (
            datetime.fromtimestamp(ftime, timezone.utc).isoformat() if ftime else None
        ),
        "forecast_raw": latest or None,
        "p_community": community_p(q) if is_binary else None,
        "n_forecasters": q.get("nr_forecasters"),
        "resolution": q.get("resolution"),
        "y": norm_y(q.get("resolution"), q.get("type")),
        "actual_resolve_time": q.get("actual_resolve_time"),
    }


def iter_post_stubs(tid):
    """Yield stubs across ALL pages of a tournament listing."""
    url = BASE + "/posts/"
    params = {"tournaments": [tid], "limit": 100}
    while url:
        page = get(url, params=params)
        params = None  # "next" is an absolute URL; params only on page 1
        for stub in page.get("results", []):
            yield stub
        url = page.get("next")


records = {}
skipped = []
for tid in TOURNAMENTS:
    n_seen = 0
    for stub in iter_post_stubs(tid):
        if "question" not in stub and "group_of_questions" not in stub:
            continue
        n_seen += 1
        try:
            post = get(BASE + "/posts/%s/" % stub["id"])
        except PostFetchError as e:
            skipped.append(stub.get("id"))
            print("  SKIP post %s: %s" % (stub.get("id"), e))
            continue
        if "question" in post:
            rec = question_record(post["question"], post, tid)
            records[str(rec["question_id"])] = rec
        else:
            for q in post["group_of_questions"].get("questions", []):
                rec = question_record(q, post, tid)
                records[str(rec["question_id"])] = rec
    print("tournament %s: %d posts scanned" % (tid, n_seen))

old = {}
if os.path.exists(LEDGER):
    old = json.load(open(LEDGER))
for qid, rec in records.items():
    rec["first_seen_at"] = old.get(qid, {}).get("first_seen_at", now_iso())
    rec["last_updated_at"] = now_iso()
    old[qid] = rec

os.makedirs("assay", exist_ok=True)
with open(LEDGER + ".tmp", "w") as f:
    json.dump(old, f, indent=2, sort_keys=True)
os.replace(LEDGER + ".tmp", LEDGER)

npred = sum(1 for r in old.values() if r["p"] is not None)
nres = sum(1 for r in old.values() if r["y"] is not None)
ncp = sum(1 for r in old.values() if r["p_community"] is not None)
print("ledger: %d questions | with forecast: %d | with community p: %d | resolved: %d | skipped posts: %d"
      % (len(old), npred, ncp, nres, len(skipped)))
for r in sorted(old.values(), key=lambda x: x["question_id"] or 0):
    print("  q%s | %s | p=%s | cp=%s | y=%s | %s" % (
        r["question_id"], r["status"], r["p"], r["p_community"], r["y"],
        (r["title"] or "")[:45]))
