const chatLog = document.getElementById("chat-log");
const composer = document.getElementById("composer");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");
const suggestionList = document.getElementById("suggestion-list");
const resetBtn = document.getElementById("reset-btn");
const lastIntentEl = document.getElementById("last-intent");
const lastConfidenceEl = document.getElementById("last-confidence");

let sessionId = localStorage.getItem("scb_session_id") || null;

function appendMessage(text, sender, meta) {
  const msg = document.createElement("div");
  msg.className = `msg ${sender}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = sender === "bot" ? "🤖" : "🙂";

  const bubbleWrap = document.createElement("div");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  bubbleWrap.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "msg-meta";
    metaEl.textContent = `intent: ${meta.intent} · confidence: ${(meta.confidence * 100).toFixed(0)}%`;
    bubbleWrap.appendChild(metaEl);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubbleWrap);
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendMessage(text) {
  if (!text.trim()) return;
  appendMessage(text, "user");
  userInput.value = "";
  typingIndicator.hidden = false;
  chatLog.scrollTop = chatLog.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    const data = await res.json();

    typingIndicator.hidden = true;

    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("scb_session_id", sessionId);
    }

    appendMessage(data.reply, "bot", { intent: data.intent, confidence: data.confidence });
    lastIntentEl.textContent = data.intent;
    lastConfidenceEl.textContent = `${(data.confidence * 100).toFixed(0)}%`;
  } catch (err) {
    typingIndicator.hidden = true;
    appendMessage("Sorry, something went wrong reaching the server. Please try again.", "bot");
    console.error(err);
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(userInput.value);
});

suggestionList.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  sendMessage(li.dataset.msg);
});

resetBtn.addEventListener("click", () => {
  sessionId = null;
  localStorage.removeItem("scb_session_id");
  chatLog.innerHTML = "";
  appendMessage(
    "Conversation reset. Hi again! What can I help you with today?",
    "bot"
  );
  lastIntentEl.textContent = "—";
  lastConfidenceEl.textContent = "—";
});
