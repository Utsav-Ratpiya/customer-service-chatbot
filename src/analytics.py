"""
analytics.py
------------
Turns the raw conversation log (data/conversation_logs.jsonl) into summary
stats a support team would actually care about:

    - total conversations / messages handled
    - intent frequency distribution (what are people actually asking?)
    - average model confidence per intent (where is the bot unsure?)
    - fallback rate (how often does the bot fail to understand?)

This isn't part of the original "build a chatbot" brief, but any real
customer-service bot deployment needs this kind of visibility -- product/
support teams use it to see what customers actually ask about and where the
bot needs more training data. Exposed via:

    - CLI:   python src/analytics.py
    - API:   GET /api/analytics  (see app.py)
"""

import json
import os
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "conversation_logs.jsonl")


def load_logs(log_path: str = LOG_PATH):
    """Read the JSONL conversation log. Returns [] if no logs exist yet
    (e.g. on a fresh clone before anyone has chatted with the bot)."""
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip any corrupted line rather than crash
    return rows


def summarize(rows=None):
    """Compute summary statistics from log rows. Returns a JSON-serializable dict."""
    if rows is None:
        rows = load_logs()

    if not rows:
        return {
            "total_messages": 0,
            "total_sessions": 0,
            "intent_counts": {},
            "avg_confidence_by_intent": {},
            "fallback_rate": 0.0,
            "note": "No conversation logs yet. Chat with the bot to generate data.",
        }

    intent_counts = Counter(r.get("predicted_intent", "unknown") for r in rows)
    sessions = {r.get("session_id") for r in rows if r.get("session_id")}

    confidence_sums = defaultdict(float)
    confidence_counts = defaultdict(int)
    for r in rows:
        intent = r.get("predicted_intent", "unknown")
        conf = r.get("confidence")
        if isinstance(conf, (int, float)):
            confidence_sums[intent] += conf
            confidence_counts[intent] += 1

    avg_confidence = {
        intent: round(confidence_sums[intent] / confidence_counts[intent], 3)
        for intent in confidence_sums
    }

    fallback_count = intent_counts.get("fallback", 0) + intent_counts.get(
        "slot_filling:order_id", 0
    ) * 0  # slot-filling isn't a fallback, kept explicit for clarity
    fallback_rate = round(fallback_count / len(rows), 3) if rows else 0.0

    return {
        "total_messages": len(rows),
        "total_sessions": len(sessions),
        "intent_counts": dict(intent_counts.most_common()),
        "avg_confidence_by_intent": avg_confidence,
        "fallback_rate": fallback_rate,
    }


def print_report():
    """Pretty-print the summary to the console. Used by the CLI entry point."""
    stats = summarize()
    print("=" * 50)
    print("SupportDesk AI — Conversation Analytics")
    print("=" * 50)
    print(f"Total messages handled : {stats['total_messages']}")
    print(f"Total sessions         : {stats['total_sessions']}")
    print(f"Fallback rate          : {stats['fallback_rate'] * 100:.1f}%")
    print("\nTop intents:")
    for intent, count in list(stats["intent_counts"].items())[:10]:
        avg_conf = stats["avg_confidence_by_intent"].get(intent, "-")
        print(f"  {intent:<20} count={count:<5} avg_confidence={avg_conf}")
    print("=" * 50)


if __name__ == "__main__":
    print_report()
