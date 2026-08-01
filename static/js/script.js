const chatLog = document.getElementById("chat-log");
const composer = document.getElementById("composer");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");
const suggestionList = document.getElementById("suggestion-list");
const resetBtn = document.getElementById("reset-btn");
const lastIntentEl = document.getElementById("last-intent");
const lastConfidenceEl = document.getElementById("last-confidence");

let sessionId = localStorage.getItem("scb_session_id") || null;

// Safe Markdown-like formatter for bold, italic and code snippets
function formatMarkdown(text) {
  if (!text) return "";
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  
  // Replace **bold** with strong
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Replace *italic* with em
  escaped = escaped.replace(/\*(.*?)\*/g, "<em>$1</em>");
  // Replace `code` with code tags
  escaped = escaped.replace(/`(.*?)`/g, "<code>$1</code>");
  
  return escaped;
}

function appendMessage(text, sender, meta) {
  const msg = document.createElement("div");
  msg.className = `msg ${sender}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = sender === "bot" ? "🤖" : "🙂";

  const bubbleWrap = document.createElement("div");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMarkdown(text);
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
    
    if (lastIntentEl) lastIntentEl.textContent = data.intent || "—";
    if (lastConfidenceEl) lastConfidenceEl.textContent = data.confidence ? `${(data.confidence * 100).toFixed(0)}%` : "—";
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
  if (lastIntentEl) lastIntentEl.textContent = "—";
  if (lastConfidenceEl) lastConfidenceEl.textContent = "—";
});


// -------------------------------------------------------------
// Admin & Dashboard Navigation and Control
// -------------------------------------------------------------
const tabChatBtn = document.getElementById("tab-chat-btn");
const tabDashboardBtn = document.getElementById("tab-dashboard-btn");
const chatPanelView = document.getElementById("chat-panel-view");
const dashboardPanelView = document.getElementById("dashboard-panel-view");
const sidebarSuggestions = document.getElementById("sidebar-suggestions");
const sidebarStatus = document.getElementById("sidebar-status");

if (tabChatBtn && tabDashboardBtn) {
  tabChatBtn.addEventListener("click", () => {
    tabChatBtn.classList.add("active");
    tabDashboardBtn.classList.remove("active");
    chatPanelView.classList.add("active");
    chatPanelView.style.display = "flex";
    dashboardPanelView.classList.remove("active");
    dashboardPanelView.style.display = "none";
    if (sidebarSuggestions) sidebarSuggestions.style.display = "block";
    if (sidebarStatus) sidebarStatus.style.display = "block";
  });

  tabDashboardBtn.addEventListener("click", () => {
    tabDashboardBtn.classList.add("active");
    tabChatBtn.classList.remove("active");
    dashboardPanelView.classList.add("active");
    dashboardPanelView.style.display = "flex";
    chatPanelView.classList.remove("active");
    chatPanelView.style.display = "none";
    if (sidebarSuggestions) sidebarSuggestions.style.display = "none";
    if (sidebarStatus) sidebarStatus.style.display = "none";
    
    // Load fresh statistics & database content
    loadDashboardStats();
    loadMockOrders();
  });
}

// Fetch analytics and load UI components
async function loadDashboardStats() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();

    document.getElementById("stat-total-messages").textContent = data.total_messages || 0;
    document.getElementById("stat-total-sessions").textContent = data.total_sessions || 0;
    document.getElementById("stat-fallback-rate").textContent = `${((data.fallback_rate || 0) * 100).toFixed(1)}%`;

    // 1. Populate Intent Frequencies List
    const intentsList = document.getElementById("stat-intents-list");
    intentsList.innerHTML = "";
    
    const counts = data.intent_counts || {};
    const confidences = data.avg_confidence_by_intent || {};
    const total = data.total_messages || 1;

    const entries = Object.entries(counts);
    if (entries.length === 0) {
      intentsList.innerHTML = '<p class="empty-text">No intent data logged yet.</p>';
    } else {
      entries.forEach(([intent, count]) => {
        const pct = ((count / total) * 100).toFixed(0);
        const confVal = confidences[intent];
        const confStr = confVal !== undefined ? `${(confVal * 100).toFixed(0)}%` : "N/A";
        
        const row = document.createElement("div");
        row.className = "intent-row";
        row.innerHTML = `
          <div class="intent-meta">
            <span class="intent-name">${intent}</span>
            <span class="intent-stats">count: ${count} (${pct}%) · confidence: ${confStr}</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: ${pct}%"></div>
          </div>
        `;
        intentsList.appendChild(row);
      });
    }

    // 2. Populate Conversations Audit List
    const auditList = document.getElementById("conversations-audit-list");
    auditList.innerHTML = "";
    const logs = data.recent_logs || [];
    
    if (logs.length === 0) {
      auditList.innerHTML = '<tr><td colspan="6" class="empty-text">No audits logged yet. Start chatting to register events!</td></tr>';
    } else {
      logs.forEach(log => {
        const tr = document.createElement("tr");
        
        let dateStr = "-";
        if (log.timestamp) {
          const date = new Date(log.timestamp);
          dateStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        
        const confVal = log.confidence;
        const confidencePct = confVal !== undefined ? `${(confVal * 100).toFixed(0)}%` : "-";
        
        tr.innerHTML = `
          <td>${dateStr}</td>
          <td title="${log.session_id || 'unknown'}"><code>${log.session_id || 'unknown'}</code></td>
          <td title="${log.message}">${log.message}</td>
          <td><span class="badge">${log.predicted_intent || "-"}</span></td>
          <td><strong>${confidencePct}</strong></td>
          <td title="${log.reply}">${log.reply}</td>
        `;
        auditList.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Error loading dashboard stats:", err);
  }
}

// Load Mock Database Table
async function loadMockOrders() {
  try {
    const res = await fetch("/api/orders");
    const orders = await res.json();
    
    const tbody = document.getElementById("db-orders-list");
    tbody.innerHTML = "";

    const entries = Object.entries(orders);
    if (entries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="empty-text">No mock orders found.</td></tr>';
    } else {
      entries.forEach(([id, info]) => {
        const tr = document.createElement("tr");
        
        let statusBadge = "badge";
        const statusClean = (info.status || "").toLowerCase();
        if (statusClean === "delivered") statusBadge = "badge badge-success";
        else if (statusClean === "shipped") statusBadge = "badge badge-info";
        else if (statusClean === "cancelled") statusBadge = "badge badge-active";

        tr.innerHTML = `
          <td><strong>${id}</strong></td>
          <td><span class="${statusBadge}">${info.status}</span></td>
          <td>${info.carrier}</td>
          <td>${info.eta}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Error loading mock database records:", err);
  }
}

// Save or Update Mock Order
const orderForm = document.getElementById("db-order-form");
const orderFormStatus = document.getElementById("order-form-status");

if (orderForm) {
  orderForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("db-order-id").value;
    const status = document.getElementById("db-order-status").value;
    const carrier = document.getElementById("db-order-carrier").value;
    const eta = document.getElementById("db-order-eta").value;

    try {
      const res = await fetch("/api/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order_id: id, status, carrier, eta })
      });
      const data = await res.json();
      
      if (data.status === "success") {
        orderFormStatus.textContent = "Order record saved successfully!";
        orderFormStatus.style.color = "#10b981";
        orderForm.reset();
        loadMockOrders();
        setTimeout(() => { orderFormStatus.textContent = ""; }, 3500);
      } else {
        orderFormStatus.textContent = data.error || "Failed to save order.";
        orderFormStatus.style.color = "#ef4444";
      }
    } catch (err) {
      orderFormStatus.textContent = "Error communicating with server.";
      orderFormStatus.style.color = "#ef4444";
    }
  });
}

// Retrain Model
const retrainBtn = document.getElementById("retrain-btn");
const retrainStatus = document.getElementById("retrain-status");

if (retrainBtn) {
  retrainBtn.addEventListener("click", async () => {
    retrainBtn.disabled = true;
    retrainStatus.textContent = "Training NLP model and loading dynamically...";
    retrainStatus.style.color = "#3b82f6";

    try {
      const res = await fetch("/api/train", { method: "POST" });
      const data = await res.json();

      if (data.status === "success") {
        retrainStatus.textContent = "Engine trained & reloaded successfully!";
        retrainStatus.style.color = "#10b981";
      } else {
        retrainStatus.textContent = data.message || "Failed to train classifier.";
        retrainStatus.style.color = "#ef4444";
      }
    } catch (err) {
      retrainStatus.textContent = "Network error while retraining.";
      retrainStatus.style.color = "#ef4444";
    } finally {
      retrainBtn.disabled = false;
      setTimeout(() => { retrainStatus.textContent = ""; }, 6000);
    }
  });
}

// Dashboard Refresh Button
const refreshDashboardBtn = document.getElementById("refresh-dashboard-btn");
if (refreshDashboardBtn) {
  refreshDashboardBtn.addEventListener("click", () => {
    loadDashboardStats();
    loadMockOrders();
  });
}
