const $ = (s) => document.querySelector(s);
let currentSlug = null;

function md(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

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

async function loadChannels() {
  const d = await (await fetch("/api/channels")).json();
  const menu = $("#channel-menu");
  menu.innerHTML = (d.channels || [])
    .map((c) => `<button data-slug="${c}">${c}</button>`)
    .join("");
  menu.querySelectorAll("[data-slug]").forEach((b) => {
    b.onclick = () => { setChannel(b.dataset.slug); menu.hidden = true; };
  });
  if (d.channels && d.channels.length) setChannel(d.channels[0]);
}

function setChannel(slug) {
  currentSlug = slug;
  $("#channel-label").textContent = slug;
}

$("#channel-btn").onclick = () => { $("#channel-menu").hidden = !$("#channel-menu").hidden; };

function addMessage(role) {
  document.body.classList.add("chatting");
  const el = document.createElement("div");
  el.className = "msg " + role;
  $("#messages").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
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
      body: JSON.stringify({ slug: currentSlug, question }),
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
      if (ev.type === "status") {
        status.textContent = ev.text;
      } else if (ev.type === "token") {
        status.textContent = "";
        answer += ev.text;
        body.innerHTML = md(answer);
        enhanceCode(body);
      } else if (ev.type === "citations") {
        if (ev.citations && ev.citations.length) {
          const src = document.createElement("div");
          src.className = "sources";
          src.innerHTML = "<b>Sources</b>" + ev.citations.map(
            (c) => `<a href="${c.url}" target="_blank">${c.title} @ ${Math.floor(c.start)}s</a>`
          ).join("");
          bot.appendChild(src);
        }
      } else if (ev.type === "error") {
        status.textContent = ev.text;
      }
      bot.scrollIntoView({ block: "end" });
    }
  }
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
document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => ask(chip.textContent);
});

loadChannels();
