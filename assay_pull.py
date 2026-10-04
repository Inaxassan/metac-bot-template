"""Assay ledger puller: reads ASSAY-Bot's Metaculus forecasts and
resolutions into assay/ledger.json. Runs in GitHub Actions.
Token never leaves GitHub secrets; output is public data."""

import json
import os
import time
from datetime import datetime, timezone

import requests

TOKEN = os.environ["METACULUS_TOKEN"]
H = {"Authorization": "Token " + TOKEN}
BASE = "https://www.metaculus.com/api"
TOURNAMENTS = ["33121", "minibench"]
LEDGER = "assay/ledger.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get(url, params=None):
    r = requests.get(url, params=params, headers=H, timeout=60)
    r.raise_for_status()
    time.sleep(1.0)
    return r.json()


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


def question_record(q, post, tournament):
    latest = (q.get("my_forecasts") or {}).get("latest") or {}
    ftime = latest.get("start_time")
    p = latest.get("probability_yes") if q.get("type") == "binary" else None
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
        "p": p,
        "forecast_time": (
            datetime.fromtimestamp(ftime, timezone.utc).isoformat() if ftime else None
        ),
        "forecast_raw": latest or None,
        "resolution": q.get("resolution"),
        "y": norm_y(q.get("resolution"), q.get("type")),
        "actual_resolve_time": q.get("actual_resolve_time"),
    }


records = {}
for tid in TOURNAMENTS:
    listing = get(BASE + "/posts/", {"tournaments": [tid], "limit": 100})
    for stub in listing.get("results", []):
        if "question" not in stub and "group_of_questions" not in stub:
            continue
        post = get(BASE + "/posts/%s/" % stub["id"])
        if "question" in post:
            rec = question_record(post["question"], post, tid)
            records[str(rec["question_id"])] = rec
        else:
            for q in post["group_of_questions"].get("questions", []):
                rec = question_record(q, post, tid)
                records[str(rec["question_id"])] = rec

old = {}
if os.path.exists(LEDGER):
    old = json.load(open(LEDGER))
for qid, rec in records.items():
    rec["first_seen_at"] = old.get(qid, {}).get("first_seen_at", now_iso())
    rec["last_updated_at"] = now_iso()
    old[qid] = rec

os.makedirs("assay", exist_ok=True)
json.dump(old, open(LEDGER, "w"), indent=2, sort_keys=True)

npred = sum(1 for r in old.values() if r["p"] is not None)
nres = sum(1 for r in old.values() if r["y"] is not None)
print("ledger: %d questions | with forecast: %d | resolved: %d" % (len(old), npred, nres))
for r in sorted(old.values(), key=lambda x: x["question_id"] or 0):
    print("  q%s | %s | p=%s | y=%s | %s" % (
        r["question_id"], r["status"], r["p"], r["y"], (r["title"] or "")[:50]))
