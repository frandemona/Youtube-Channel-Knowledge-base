const $ = (s) => document.querySelector(s);
let currentSlug = null;
let conversationId = null;

function md(text) { return DOMPurify.sanitize(marked.parse(text || "")); }

function enhanceCode(container) {
  container.querySelectorAll("pre > code").forEach((code) => {
    if (code.dataset.enhanced) return;
    code.dataset.enhanced = "1";
    try { hljs.highlightElement(code); } catch (e) {}
    const pre = code.parentElement;
    const lang = (code.className.match(/language-(\w+)/) || [, "code"])[1];
    const header = document.createElement("div");
    header.className = "code-header";
    header.innerHTML = `<span>${lang}</span><button class="copy">Copy code</button>`;
    header.querySelector(".copy").onclick = () => {
      navigator.clipboard.writeText(code.innerText);
      header.querySelector(".copy").textContent = "Copied";
      setTimeout(() => (header.querySelector(".copy").textContent = "Copy code"), 1200);
    };
    pre.prepend(header);
  });
}

function citationsHtml(citations) {
  if (!citations || !citations.length) return "";
  return DOMPurify.sanitize('<div class="sources"><b>Sources</b>' + citations.map(
    (c) => `<a href="${c.url}" target="_blank" rel="noopener noreferrer">${c.title} @ ${Math.floor(c.start)}s</a>`
  ).join("") + "</div>");
}

async function loadChannels() {
  const d = await (await fetch("/api/channels")).json();
  const menu = $("#channel-menu");
  menu.innerHTML = (d.channels || []).map((c) => `<button data-slug="${c}">${c}</button>`).join("");
  menu.querySelectorAll("[data-slug]").forEach((b) => {
    b.onclick = () => { setChannel(b.dataset.slug); menu.hidden = true; };
  });
  if (d.channels && d.channels.length && !currentSlug) setChannel(d.channels[0]);
}

function setChannel(slug) {
  currentSlug = slug;
  $("#channel-label").textContent = slug || "";
}

function lockChannel(locked) {
  $("#channel-btn").disabled = locked;
  $("#channel-btn").style.opacity = locked ? "0.4" : "1";
  if (locked) $("#channel-menu").hidden = true;
}

$("#channel-btn").onclick = () => {
  if ($("#channel-btn").disabled) return;
  $("#channel-menu").hidden = !$("#channel-menu").hidden;
};

async function loadConversations() {
  const d = await (await fetch("/api/conversations")).json();
  const list = $("#conversations");
  list.innerHTML = "";
  (d.conversations || []).forEach((c) => {
    const row = document.createElement("div");
    row.className = "conv" + (c.id === conversationId ? " active" : "");
    row.innerHTML = `<span class="conv-title">${DOMPurify.sanitize(c.title || "New chat")}</span>` +
                    `<span class="conv-badge">${DOMPurify.sanitize(c.slug)}</span>` +
                    `<button class="conv-del" title="Delete">×</button>`;
    const titleEl = row.querySelector(".conv-title");
    titleEl.title = c.title || "New chat"; // native tooltip shows the full, overflowing title
    titleEl.onclick = () => openConversation(c.id);
    row.querySelector(".conv-badge").onclick = () => openConversation(c.id);
    row.querySelector(".conv-del").onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === conversationId) newChat();
      loadConversations();
    };
    list.appendChild(row);
  });
}

function newChat() {
  conversationId = null;
  $("#messages").innerHTML = "";
  document.body.classList.remove("chatting");
  lockChannel(false);
  loadConversations();
}
$("#new-chat").onclick = newChat;

function addMessage(role) {
  document.body.classList.add("chatting");
  const el = document.createElement("div");
  el.className = "msg " + role;
  $("#messages").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function renderAssistant(content, citations) {
  const bot = addMessage("assistant");
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = md(content);
  bot.appendChild(body);
  enhanceCode(body);
  if (citations && citations.length) {
    const src = document.createElement("div");
    src.innerHTML = citationsHtml(citations);
    bot.appendChild(src.firstChild);
  }
}

async function openConversation(id) {
  const conv = await (await fetch(`/api/conversations/${id}`)).json();
  conversationId = id;
  setChannel(conv.slug);
  lockChannel(true);
  $("#messages").innerHTML = "";
  document.body.classList.add("chatting");
  conv.messages.forEach((m) => {
    if (m.role === "user") addMessage("user").textContent = m.content;
    else renderAssistant(m.content, m.citations);
  });
  loadConversations();
}

async function ask(question) {
  if (!question.trim() || !currentSlug) return;
  addMessage("user").textContent = question;
  const bot = addMessage("assistant");
  const status = document.createElement("div");
  status.className = "status";
  const body = document.createElement("div");
  body.className = "body";
  bot.append(status, body);

  let answer = "";
  let res;
  try {
    res = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: currentSlug, question, conversation_id: conversationId }),
    });
  } catch (e) { status.textContent = "Network error: " + e; return; }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.replace(/^data: /, "").trim();
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); } catch (e) { continue; }
      if (ev.type === "conversation") {
        conversationId = ev.id;
        lockChannel(true);
        loadConversations();
      } else if (ev.type === "status") {
        status.textContent = ev.text;
      } else if (ev.type === "token") {
        status.textContent = "";
        answer += ev.text;
        body.innerHTML = md(answer);
      } else if (ev.type === "citations") {
        if (ev.citations && ev.citations.length) {
          const src = document.createElement("div");
          src.innerHTML = citationsHtml(ev.citations);
          bot.appendChild(src.firstChild);
        }
      } else if (ev.type === "title") {
        loadConversations();
      } else if (ev.type === "error") {
        status.textContent = ev.text;
      }
      bot.scrollIntoView({ block: "end" });
    }
  }
  body.innerHTML = md(answer);
  enhanceCode(body);
}

function send() {
  const q = $("#q").value;
  $("#q").value = "";
  ask(q);
}
$("#send").onclick = send;
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.querySelectorAll(".chip").forEach((chip) => { chip.onclick = () => ask(chip.textContent); });

loadChannels();
loadConversations();
