"""
test_analytics.py
------------------
Unit tests for src/analytics.py. Uses a temporary log file so tests never
touch the real data/conversation_logs.jsonl and can run in any order.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from analytics import load_logs, summarize


SAMPLE_ROWS = [
    {"session_id": "s1", "predicted_intent": "greeting", "confidence": 0.9},
    {"session_id": "s1", "predicted_intent": "order_status", "confidence": 0.8},
    {"session_id": "s2", "predicted_intent": "fallback", "confidence": 0.1},
    {"session_id": "s2", "predicted_intent": "order_status", "confidence": 0.6},
]


def _write_temp_log(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class TestAnalytics(unittest.TestCase):
    def test_load_logs_missing_file_returns_empty_list(self):
        self.assertEqual(load_logs("/nonexistent/path.jsonl"), [])

    def test_summarize_empty_logs(self):
        stats = summarize([])
        self.assertEqual(stats["total_messages"], 0)
        self.assertIn("note", stats)

    def test_summarize_counts_and_sessions(self):
        stats = summarize(SAMPLE_ROWS)
        self.assertEqual(stats["total_messages"], 4)
        self.assertEqual(stats["total_sessions"], 2)
        self.assertEqual(stats["intent_counts"]["order_status"], 2)

    def test_summarize_average_confidence(self):
        stats = summarize(SAMPLE_ROWS)
        self.assertAlmostEqual(stats["avg_confidence_by_intent"]["order_status"], 0.7)

    def test_summarize_fallback_rate(self):
        stats = summarize(SAMPLE_ROWS)
        self.assertAlmostEqual(stats["fallback_rate"], 0.25)

    def test_load_logs_from_real_file(self):
        path = _write_temp_log(SAMPLE_ROWS)
        try:
            rows = load_logs(path)
            self.assertEqual(len(rows), 4)
        finally:
            os.remove(path)

    def test_load_logs_skips_corrupted_lines(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(SAMPLE_ROWS[0]) + "\n")
            f.write("{not valid json\n")
            f.write(json.dumps(SAMPLE_ROWS[1]) + "\n")
        try:
            rows = load_logs(path)
            self.assertEqual(len(rows), 2)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
