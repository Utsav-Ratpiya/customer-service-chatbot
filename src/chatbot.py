"""
chatbot.py
----------
Core hybrid chatbot engine.

Design (industry-style hybrid, not purely one or the other):
    1. NLP LAYER   - a trained TF-IDF + Logistic Regression model predicts
                      the user's intent from free text.
    2. RULE LAYER  - regex-based entity extraction (order IDs, emails) and
                      hand-written business rules for what happens per
                      intent (e.g. "order_status" needs an order ID before
                      it can give a real answer -> slot filling).
    3. CONTEXT     - a tiny per-session state machine remembers whether the
                      bot is waiting on a follow-up answer (e.g. order ID)
                      and short-circuits the NLP layer while it does.
    4. FALLBACK    - if the model's confidence is below a threshold, the
                      bot asks for clarification instead of guessing.

This mirrors how many real production support bots are built: an ML intent
layer for coverage/generalization, wrapped in deterministic business rules
for anything that touches real data (orders, payments, refunds).
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime, timezone

import joblib

sys.path.append(os.path.dirname(__file__))
from nlp_utils import preprocess_for_vectorizer, extract_order_id, extract_email

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
LOG_PATH = os.path.join(BASE_DIR, "data", "conversation_logs.jsonl")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.35))

# Intents that require an order ID before the bot can respond meaningfully.
ORDER_ID_INTENTS = {"order_status", "cancel_order", "refund_status"}

# A tiny mock "database" so the demo produces believable, varied answers
# without needing a real backend. Swap this for a real order-service call
# in production.
MOCK_ORDER_DB = {
    "ORD12345": {"status": "Shipped", "eta": "2 business days", "carrier": "BlueDart"},
    "ORD67890": {"status": "Processing", "eta": "not yet dispatched", "carrier": "-"},
    "ORD11111": {"status": "Delivered", "eta": "delivered on 2026-07-20", "carrier": "FedEx"},
}


class ChatbotEngine:
    """Stateless w.r.t. the model (loaded once); per-user conversation
    state is tracked externally via `session_state` dicts so the same
    engine instance can serve many concurrent users (e.g. in a Flask app)."""

    def __init__(self, data_path=DATA_PATH, model_path=MODEL_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)

        self.responses = {
            intent["tag"]: intent["responses"] for intent in self.intents_data["intents"]
        }
        self.context_set = {
            intent["tag"]: intent.get("context_set") for intent in self.intents_data["intents"]
        }

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python src/train_model.py` first."
            )
        self.model = joblib.load(model_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def new_session(self):
        """Create a fresh conversation state for a new user/session."""
        return {"session_id": str(uuid.uuid4()), "awaiting": None, "history": []}

    def get_response(self, message: str, session_state: dict):
        """Main entry point. Returns (reply_text, updated_session_state, meta)."""
        message = message.strip()
        if not message:
            return "Could you type your question?", session_state, {"intent": None, "confidence": 0.0}

        # --- 1. RULE LAYER: are we mid-conversation, waiting on a slot? ---
        if session_state.get("awaiting") == "order_id":
            order_id = extract_order_id(message)
            reply = self._handle_order_id_followup(order_id, session_state)
            session_state["awaiting"] = None
            self._log(message, "slot_filling:order_id", 1.0, reply, session_state)
            return reply, session_state, {"intent": "slot_filling", "confidence": 1.0}

        # --- 2. NLP LAYER: classify intent ---
        cleaned = preprocess_for_vectorizer(message)
        probs = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_
        best_idx = probs.argmax()
        intent = classes[best_idx]
        confidence = float(probs[best_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            intent = "fallback"

        # --- 3. RULE LAYER: business logic per intent ---
        reply = self._build_reply(intent, message, session_state)

        self._log(message, intent, confidence, reply, session_state)
        return reply, session_state, {"intent": intent, "confidence": round(confidence, 3)}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_reply(self, intent, message, session_state):
        base_reply = random.choice(self.responses.get(intent, self.responses["fallback"]))

        # If this intent needs an order ID, check whether the user already
        # supplied one in the same message (e.g. "track order ORD12345").
        if intent in ORDER_ID_INTENTS:
            order_id = extract_order_id(message)
            if order_id:
                return self._handle_order_id_followup(order_id, session_state, intent=intent)
            else:
                session_state["awaiting"] = "order_id"
                return base_reply

        return base_reply

    def _handle_order_id_followup(self, order_id, session_state, intent="order_status"):
        if not order_id:
            session_state["awaiting"] = "order_id"
            return ("I couldn't find a valid order ID in that message. "
                    "It usually looks like ORD12345 — could you double check and resend it?")

        order = MOCK_ORDER_DB.get(order_id.replace("ORDER", "ORD"))
        if not order:
            return (f"I couldn't find any order with ID {order_id} in our system. "
                     "Could you double-check the ID, or would you like me to connect you to an agent?")

        if intent == "cancel_order":
            if order["status"] == "Processing":
                return (f"Order {order_id} is still processing, so it's eligible for cancellation. "
                         "I've submitted the cancellation request — you'll get a confirmation email shortly.")
            else:
                return (f"Order {order_id} has already been {order['status'].lower()}, "
                         "so it can no longer be cancelled. You're welcome to request a return instead.")

        if intent == "refund_status":
            return (f"Order {order_id} is currently marked '{order['status']}'. "
                     "If a return has been received, refunds typically post within 5-7 business days.")

        # default: order_status
        return (f"Order {order_id} is currently **{order['status']}** "
                 f"(carrier: {order['carrier']}, ETA: {order['eta']}).")

    def _log(self, message, intent, confidence, reply, session_state):
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_state.get("session_id"),
                    "message": message,
                    "predicted_intent": intent,
                    "confidence": confidence,
                    "reply": reply,
                }) + "\n")
        except OSError:
            pass  # logging must never break the chat experience


if __name__ == "__main__":
    # Simple CLI smoke test
    bot = ChatbotEngine()
    state = bot.new_session()
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        reply, state, meta = bot.get_response(user_input, state)
        print(f"Bot [{meta['intent']} | conf={meta['confidence']}]: {reply}")
