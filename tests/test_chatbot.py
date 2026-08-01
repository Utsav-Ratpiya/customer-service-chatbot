"""
test_chatbot.py
----------------
Unit tests for the chatbot's NLP utilities and end-to-end conversation
behaviour. Run with:

    pytest tests/ -v

or, without pytest installed:

    python -m unittest tests.test_chatbot -v
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from nlp_utils import clean_text, tokenize, extract_order_id, extract_email, preprocess_for_vectorizer, spell_correct_word
from chatbot import ChatbotEngine


class TestNlpUtils(unittest.TestCase):
    def test_clean_text_lowercases_and_strips_punctuation(self):
        self.assertEqual(clean_text("Hello, World!!"), "hello world")

    def test_clean_text_expands_contractions(self):
        cleaned = clean_text("I can't find my order")
        self.assertIn("not", cleaned)

    def test_tokenize_removes_stopwords(self):
        tokens = tokenize("What is the price of the product")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)

    def test_tokenize_keeps_negations(self):
        tokens = tokenize("my order has not arrived")
        self.assertIn("not", tokens)

    def test_extract_order_id_variants(self):
        self.assertEqual(extract_order_id("track ORD12345 please"), "ORD12345")
        self.assertEqual(extract_order_id("my order is #98765"), "98765")
        self.assertEqual(extract_order_id("no id here"), None)

    def test_extract_email(self):
        self.assertEqual(extract_email("reach me at test.user@example.com"), "test.user@example.com")
        self.assertIsNone(extract_email("no email here"))

    def test_preprocess_for_vectorizer_returns_string(self):
        result = preprocess_for_vectorizer("Where IS my order??")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_spell_correction(self):
        self.assertEqual(spell_correct_word("shippng"), "shipping")
        self.assertEqual(spell_correct_word("refunde"), "refund")
        self.assertEqual(spell_correct_word("normal"), "normal")


class TestChatbotEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = ChatbotEngine()

    def setUp(self):
        self.state = self.bot.new_session()

    def test_greeting_intent(self):
        reply, state, meta = self.bot.get_response("hello there", self.state)
        self.assertEqual(meta["intent"], "greeting")
        self.assertTrue(len(reply) > 0)

    def test_return_policy_intent(self):
        reply, state, meta = self.bot.get_response("what is your return policy", self.state)
        self.assertEqual(meta["intent"], "return_policy")
        self.assertIn("30 days", reply)

    def test_order_status_slot_filling_flow(self):
        reply1, state, meta1 = self.bot.get_response("where is my order", self.state)
        self.assertEqual(meta1["intent"], "order_status")
        self.assertEqual(state["awaiting"], "order_id")

        reply2, state, meta2 = self.bot.get_response("ORD12345", state)
        self.assertEqual(meta2["intent"], "slot_filling")
        self.assertIn("Shipped", reply2)
        self.assertIsNone(state["awaiting"])

    def test_order_status_with_id_in_same_message(self):
        reply, state, meta = self.bot.get_response("track my order ORD11111", self.state)
        self.assertEqual(meta["intent"], "order_status")
        self.assertIn("Delivered", reply)

    def test_unknown_order_id(self):
        reply, state, meta = self.bot.get_response("where is my order", self.state)
        reply2, state, meta2 = self.bot.get_response("ORD99999", state)
        self.assertIn("couldn't find", reply2)

    def test_fallback_on_gibberish(self):
        reply, state, meta = self.bot.get_response("asjdkahsdkjh qwoieqwoie", self.state)
        self.assertEqual(meta["intent"], "fallback")

    def test_empty_message_handled_gracefully(self):
        reply, state, meta = self.bot.get_response("   ", self.state)
        self.assertIsNone(meta["intent"])
        self.assertTrue(len(reply) > 0)

    def test_session_is_independent_across_users(self):
        state_a = self.bot.new_session()
        state_b = self.bot.new_session()
        self.bot.get_response("where is my order", state_a)
        self.assertEqual(state_a["awaiting"], "order_id")
        self.assertIsNone(state_b["awaiting"])

    def test_order_status_slot_filling_cancel(self):
        reply1, state, meta1 = self.bot.get_response("where is my order", self.state)
        self.assertEqual(state["awaiting"], "order_id")

        reply2, state, meta2 = self.bot.get_response("nevermind", state)
        self.assertIsNone(state["awaiting"])
        self.assertIn("cancelled", reply2.lower())

    def test_order_status_slot_filling_override_with_high_conf_intent(self):
        reply1, state, meta1 = self.bot.get_response("where is my order", self.state)
        self.assertEqual(state["awaiting"], "order_id")

        reply2, state, meta2 = self.bot.get_response("what is your return policy", state)
        self.assertIsNone(state["awaiting"])
        self.assertEqual(meta2["intent"], "return_policy")
        self.assertIn("30 days", reply2)


if __name__ == "__main__":
    unittest.main()
