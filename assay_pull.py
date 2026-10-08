"""Assay ledger puller v3.4: reads ASSAY-Bot's Metaculus forecasts, community
predictions, and resolutions into assay/ledger.json. Runs in GitHub Actions.
Token never leaves GitHub secrets; output is public data.

Design (2026-10-08), aligned with the maintained Metaculus forecasting-tools
client and the Metaculus backend source (posts/views.py, questions/services/
forecasts.py, utils/views.py), hardened by five production findings:
- Offset pagination (limit=100, offset=N); zero-new-IDs stop; page cap.
  (v2 followed "next" URLs and looped silently for 6h.)
- Never silent: every page, retry, and fetch prints with flush=True.
- Hard time budget: on exhaustion, writes a PARTIAL ledger and exits 0.
  The workflow's timeout-minutes is the outer backstop.
- Listing failures after retries fail the run (dead-man: stale ledger is the
  visible signal). Detail/explorer failures skip that item this run.
- NEVER-FORGET MERGE: a run that fails to learn a field fills it from the
  old ledger instead of erasing it (p, p_vec, p_community, n_forecasters,
  forecast_time, forecast_raw, community_raw, cp_checked_at, cp_reveal_time,
  cp_scope).
- CP CAN BE SERVER-HIDDEN: is_cp_hidden empties aggregations while unresolved
  with cp_reveal_time in the future. We record cp_reveal_time, skip fetches
  while hidden, and fetch CP once after resolution/reveal when never captured.
- CP SOURCE REDESIGN (v3.4): post detail always carries an aggregations key
  but it is EMPTY when all forecasts are bot-authored -- the backend excludes
  bots unless question.include_bots_in_aggregates (default False; proven by
  runs #14+ and telemetry none=70 hidden=0). CP now comes from
  /api/aggregation_explorer/?question_id=N[&include_bots=true], which also
  returns forecasters_count. Scope is tried default-then-bots (bots-first when
  the ledger remembers with_bots); cp_scope records WHICH CROWD the number
  describes -- a bot-inclusive aggregate is never conflated with the human
  crowd. When both scopes are empty the run logs a keys-only cp-probe line
  so shape drift is visible instead of silent.
- EMPTY-STAMP + SPLIT CAPS (v3.4.1): a question null in BOTH scopes is stamped
  cp_scope="empty" and re-watched on the CP_REFRESH cadence (6-hourly) instead
  of re-probing every run; detail backfill and explorer probes carry
  SEPARATE call caps (MAX_DETAIL / MAX_EXPLORER) so neither starves the other.
- V3.4.2: explorer calls always send aggregation_methods -- the backend rejects
  include_bots without it (HTTP 400, the run-#21 failure). And a FAILED fetch
  no longer earns the "empty" stamp: empty means a real 200 with no crowd,
  never a network/HTTP error; failures retry next run instead of writing
  a verdict without evidence.
- Ledger write is atomic (tmp file + os.replace).
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
EXPLORER = BASE + "/aggregation_explorer/"
TOURNAMENTS = [t.strip() for t in
               os.environ.get("ASSAY_TOURNAMENTS", "33121,minibench").split(",")
               if t.strip()]
LEDGER = "assay/ledger.json"
STATUSES = ["open", "closed", "resolved", "upcoming"]

PAGE_SIZE = 100
MAX_PAGES = 25          # hard stop, loop-proof: 25*100 = 2500 posts max
PACE = 1.2              # seconds between successful calls (+ jitter)
MAX_TRIES = 4           # per request before giving up
MAX_BACKOFF = 60        # seconds; Retry-After is honored but capped here
BUDGET_S = 1200         # 20 min; workflow timeout-minutes: 30 is the backstop
MAX_DETAIL = 60         # detail GETs per run (my_forecasts backfill)
MAX_EXPLORER = 120      # explorer GETs per run (CP probes); independent cap
CP_REFRESH_S = 21600    # re-fetch a question's community data at most 6-hourly
CP_TYPES = ("binary", "multiple_choice")

SESSION = requests.Session()
SESSION.headers["Authorization"] = "Token " + TOKEN
SESSION.headers["User-Agent"] = "assay-ledger/3.4.2 (github-actions)"

T0 = time.monotonic()
PARTIAL = False

# fields filled from the old ledger when a run fails to learn them anew
FILL_FROM_OLD = ("p", "p_vec", "p_community", "n_forecasters", "forecast_time",
                 "forecast_raw", "community_raw", "cp_checked_at",
                 "cp_reveal_time", "cp_scope")


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


def parse_iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


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


def my_p_vec(latest, qtype):
    """Full probability vector for multiple_choice questions."""
    if not latest or qtype != "multiple_choice":
        return None
    fv = latest.get("forecast_values")
    if fv and len(fv) > 2:
        return [float(x) for x in fv]
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
            fv = latest.get("forecast_values")
            if fv and len(fv) == 2:
                return float(fv[1])
        for key in ("community_prediction", "metaculus_prediction"):
            full = (q.get(key) or {}).get("full") or {}
            if full.get("q2") is not None:
                return float(full["q2"])
    except Exception:
        pass
    return None


def community_raw(q):
    """Compact copy of the community aggregation blob, kept for shape
    discovery -- the API has renamed fields on us before."""
    agg = q.get("aggregations") or {}
    out = {}
    for method in ("recency_weighted", "unweighted"):
        latest = (agg.get(method) or {}).get("latest")
        if latest:
            out[method] = latest
    return out or None


def n_forecasters(q):
    """Forecaster count across known key names, incl. inside aggregations.
    aggregation_explorer returns forecasters_count at top level."""
    for key in ("nr_forecasters", "forecasters_count", "forecasts_count"):
        if q.get(key) is not None:
            return q.get(key)
    try:
        agg = q.get("aggregations") or {}
        for method in ("recency_weighted", "unweighted"):
            latest = (agg.get(method) or {}).get("latest") or {}
            for key in ("forecast_count", "forecaster_count", "nr_forecasters"):
                if latest.get(key) is not None:
                    return latest.get(key)
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
        "p": my_p(latest, q.get("type")),
        "p_vec": my_p_vec(latest, q.get("type")),
        "forecast_time": (
            datetime.fromtimestamp(ftime, timezone.utc).isoformat() if ftime else None
        ),
        "forecast_raw": latest or None,
        "p_community": community_p(q) if is_binary else None,
        "community_raw": community_raw(q),
        "n_forecasters": n_forecasters(q),
        "cp_reveal_time": q.get("cp_reveal_time"),
        "cp_scope": None,
        "cp_checked_at": None,
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
        if not isinstance(page, dict):
            raise PostFetchError("malformed listing page for %s: body is %s"
                                 % (tid, type(page).__name__))
        results = page.get("results", [])
        if not isinstance(results, list):
            raise PostFetchError("malformed listing page for %s: results is %s"
                                 % (tid, type(results).__name__))
        fresh = [st for st in results if st.get("id") not in seen_post_ids]
        log("tournament %s page %d: %d posts (%d new)"
            % (tid, page_num, len(results), len(fresh)))
        for stub in fresh:
            seen_post_ids.add(stub.get("id"))
            if "question" in stub or "group_of_questions" in stub:
                yield stub
        if len(results) < PAGE_SIZE or not fresh:
            break


def questions_of(post):
    if "question" in post:
        return [post["question"]]
    return post.get("group_of_questions", {}).get("questions", [])


def cp_stale(old_rec):
    """True when community data was never fetched or is older than the refresh
    interval -- bounds explorer fetches while keeping cp fresh for scoring."""
    last = parse_iso((old_rec or {}).get("cp_checked_at"))
    if last is None:
        return True
    return (datetime.now(timezone.utc) - last).total_seconds() > CP_REFRESH_S


def cp_hidden(rec):
    """True while the backend hides the community prediction: unresolved and
    cp_reveal_time in the future (is_cp_hidden empties aggregations)."""
    if rec.get("resolution") is not None:
        return False
    reveal = parse_iso(rec.get("cp_reveal_time"))
    return reveal is not None and reveal > datetime.now(timezone.utc)


def want_cp(rec, old_rec):
    """Whether an explorer fetch for community data is worthwhile this run."""
    if rec["type"] not in CP_TYPES:
        return False
    if cp_hidden(rec):
        return False                      # server-side hidden; don't ask yet
    if rec["y"] is not None:
        # resolved/revealed: capture CP once (binary only; MC cp unused)
        if rec["type"] != "binary":
            return False
        if (old_rec or {}).get("p_community") is not None:
            return False
    scope = (old_rec or {}).get("cp_scope")
    if scope == "empty":
        return cp_stale(old_rec)          # v3.4.1: re-watch 6-hourly, not hourly
    if (old_rec or {}).get("p_community") is None and scope is None:
        return True   # pre-v3.4 record: re-probe with explorer once, stamp or no
    return cp_stale(old_rec)


def fetch_cp(rec, old_rec):
    """aggregation_explorer fetch with scope strategy. Bot-authored forecasts
    are excluded from default-scope aggregates (backend: include_bots filter),
    so 'default' empty -> retry with include_bots=true. cp_scope records which
    crowd the number describes. Returns (got_data, last_response).
    v3.4.1: both scopes empty -> cp_scope='empty' so the question is re-watched
    on the refresh cadence instead of every run.
    v3.4.2: aggregation_methods always sent (backend 400s include_bots without
    it); the 'empty' stamp requires a real response -- failures stay unverdicted
    and retry next run."""
    scope_order = ["default", "bots"]
    if (old_rec or {}).get("cp_scope") == "with_bots":
        scope_order = ["bots", "default"]      # remember where the crowd lives
    resp = None
    for scope in scope_order:
        params = {"question_id": rec["question_id"],
                  "aggregation_methods": "recency_weighted,unweighted"}
        if scope == "bots":
            params["include_bots"] = "true"
        try:
            resp = get(EXPLORER, params=params)
        except PostFetchError as e:
            log("SKIP cp q%s: %s" % (rec["question_id"], e))
            break                            # outage: don't hammer the fallback
        nf = n_forecasters(resp)
        if nf is not None:
            rec["n_forecasters"] = nf        # 0 is a truthful reading, keep it
        if rec["type"] == "binary":
            cp = community_p(resp)
            if cp is not None:
                rec["p_community"] = cp
        raw = community_raw(resp)
        if raw is not None:
            rec["community_raw"] = raw
        if rec["p_community"] is not None or raw is not None or (nf or 0) > 0:
            rec["cp_scope"] = "with_bots" if scope == "bots" else "default"
            return True, resp
    if resp is not None:
        rec["cp_scope"] = "empty"    # verdict earned: real 200, no crowd
    return False, resp               # resp None = fetch failed: no verdict


def cp_probe_log(rec, resp):
    """Keys-only shape telemetry when both scopes came back empty -- the API
    has renamed fields on us before; never leak values, only structure."""
    if resp is None:
        log("cp-probe q%s: no response (fetch skipped/failed)"
            % rec["question_id"])
        return
    agg = resp.get("aggregations") or {}
    rw_latest = (agg.get("recency_weighted") or {}).get("latest")
    log("cp-probe q%s: resp_keys=%s agg_methods=%s rw_latest=%s fc=%s bots_flag=%s"
        % (rec["question_id"], sorted(resp.keys())[:16], sorted(agg.keys()),
           str(rw_latest)[:80], resp.get("forecasters_count"),
           resp.get("include_bots_in_aggregates")))


def main():
    global PARTIAL
    old = {}
    if os.path.exists(LEDGER):
        old = json.load(open(LEDGER))

    records = {}
    skipped = []
    n_detail = 0
    n_explorer = 0
    cp_src = {"kept": 0, "explorer": 0, "none": 0, "hidden": 0, "bots": 0}

    for tid in TOURNAMENTS:
        n_seen = 0
        for stub in iter_post_stubs(tid):
            n_seen += 1
            qs = questions_of(stub)

            need_keys = any("my_forecasts" not in q for q in qs)
            if need_keys and n_detail < MAX_DETAIL and not over_budget():
                try:
                    n_detail += 1
                    detail = get(BASE + "/posts/%s/" % stub["id"],
                                 params={"with_cp": "true"})
                    dqs = {q.get("id"): q for q in questions_of(detail)}
                    qs = [{**q, **dqs[q.get("id")]} if q.get("id") in dqs else q
                          for q in qs]
                except PostFetchError as e:
                    skipped.append(stub.get("id"))
                    log("SKIP post %s: %s" % (stub.get("id"), e))

            prelim = [question_record(q, stub, tid) for q in qs]
            cp_src["hidden"] += sum(1 for r in prelim if cp_hidden(r))

            for rec in prelim:
                old_rec = old.get(str(rec["question_id"]))
                if (want_cp(rec, old_rec) and n_explorer < MAX_EXPLORER
                        and not over_budget()):
                    n_explorer += 1
                    got, resp = fetch_cp(rec, old_rec)
                    rec["cp_checked_at"] = now_iso()
                    cp_src["explorer" if got else "none"] += 1
                    if got and rec["cp_scope"] == "with_bots":
                        cp_src["bots"] += 1
                    if not got:
                        cp_probe_log(rec, resp)
                else:
                    cp_src["kept"] += 1
                records[str(rec["question_id"])] = rec
        log("tournament %s: %d posts scanned" % (tid, n_seen))
        if over_budget():
            PARTIAL = True
            log("BUDGET exhausted; writing partial ledger")
            break

    for qid, rec in records.items():
        oldr = old.get(qid, {})
        for k in FILL_FROM_OLD:
            if rec.get(k) is None and oldr.get(k) is not None:
                rec[k] = oldr[k]
        rec["first_seen_at"] = oldr.get("first_seen_at", now_iso())
        rec["last_updated_at"] = now_iso()
        old[qid] = rec

    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER + ".tmp", "w") as f:
        json.dump(old, f, indent=2, sort_keys=True)
    os.replace(LEDGER + ".tmp", LEDGER)

    npred = sum(1 for r in old.values() if r["p"] is not None)
    nres = sum(1 for r in old.values() if r["y"] is not None)
    ncp = sum(1 for r in old.values() if r["p_community"] is not None)
    nemp = sum(1 for r in old.values() if r.get("cp_scope") == "empty")
    log("ledger%s: %d questions | with forecast: %d | with community p: %d | resolved: %d | detail GETs: %d | explorer GETs: %d | empty-stamped: %d | skipped posts: %d"
        % (" (PARTIAL)" if PARTIAL else "", len(old), npred, ncp, nres,
           n_detail, n_explorer, nemp, len(skipped)))
    log("cp sources: kept=%d explorer=%d none=%d hidden=%d bots=%d"
        % (cp_src["kept"], cp_src["explorer"], cp_src["none"],
           cp_src["hidden"], cp_src["bots"]))
    for r in sorted(old.values(), key=lambda x: x["question_id"] or 0):
        log("q%s | %s | p=%s | cp=%s | y=%s | %s" % (
            r["question_id"], r["status"], r["p"], r["p_community"], r["y"],
            (r["title"] or "")[:45]))


if __name__ == "__main__":
    main()
