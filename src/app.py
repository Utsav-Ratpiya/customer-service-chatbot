"""
app.py
------
Flask web application that serves:
    - GET  /              -> chat UI (templates/index.html)
    - POST /api/chat       -> {"message": "...", "session_id": "..."} -> JSON reply
    - GET  /api/health     -> health check

Run:
    python src/app.py
Then open http://localhost:5000
"""

import os
import sys

from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.dirname(__file__))
from chatbot import ChatbotEngine, load_mock_orders, save_mock_orders
from analytics import summarize as summarize_logs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

bot = ChatbotEngine()

# In-memory session store keyed by session_id. Fine for a demo/single
# instance; swap for Redis or a DB-backed store in a real deployment.
SESSIONS = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = payload.get("message", "")
    session_id = payload.get("session_id")

    if session_id and session_id in SESSIONS:
        state = SESSIONS[session_id]
    else:
        state = bot.new_session()
        session_id = state["session_id"]
        SESSIONS[session_id] = state

    if not message.strip():
        return jsonify({"error": "message is required"}), 400

    reply, state, meta = bot.get_response(message, state)
    SESSIONS[session_id] = state

    return jsonify({
        "reply": reply,
        "session_id": session_id,
        "intent": meta["intent"],
        "confidence": meta["confidence"],
    })


@app.route("/api/analytics")
def analytics():
    """Summary stats over all logged conversations: intent frequency,
    average confidence per intent, and fallback rate. Not part of the
    original brief, but real support-bot deployments need this kind of
    visibility into what customers are actually asking."""
    return jsonify(summarize_logs())


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/orders", methods=["GET"])
def get_orders():
    return jsonify(load_mock_orders())


@app.route("/api/orders", methods=["POST"])
def update_orders():
    payload = request.get_json(force=True, silent=True) or {}
    order_id = payload.get("order_id")
    status = payload.get("status")
    carrier = payload.get("carrier", "-")
    eta = payload.get("eta", "-")
    
    if not order_id or not status:
        return jsonify({"error": "order_id and status are required"}), 400
        
    orders = load_mock_orders()
    formatted_id = order_id.strip().upper().replace(" ", "")
    orders[formatted_id] = {
        "status": status.strip(),
        "carrier": carrier.strip(),
        "eta": eta.strip()
    }
    save_mock_orders(orders)
    return jsonify({"status": "success", "orders": orders})


@app.route("/api/train", methods=["POST"])
def train_model_endpoint():
    try:
        from train_model import train as run_train
        from nlp_utils import load_vocabulary
        run_train()
        load_vocabulary()
        global bot
        bot = ChatbotEngine()
        return jsonify({"status": "success", "message": "Model retrained and loaded successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
