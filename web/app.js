/* DuplexRAG demo client: streams transcript chunks to the engine and renders its telemetry live. */
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { ws: null, turns: {}, current: null, answerText: "", answerIntents: [], queries: {},
                  lastChunkEl: null, livePill: null, micOn: false };

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  try { const t = localStorage.getItem("duplexrag-theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("btn-theme").onclick = () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("duplexrag-theme", root.dataset.theme); } catch (e) {}
  };

  // ------------------------------------------------------------------ websocket
  function connect() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    state.ws = ws;
    ws.onmessage = (m) => handle(JSON.parse(m.data));
    ws.onclose = () => { $("session-pill").textContent = "disconnected - retrying"; setTimeout(connect, 1500); };
  }
  const send = (o) => state.ws && state.ws.readyState === 1 && state.ws.send(JSON.stringify(o));

  function handle(msg) {
    if (msg.type === "session") return onSession(msg);
    if (msg.type === "event") return onEvent(msg.event);
    if (msg.type === "token") { state.answerText += msg.text; renderAnswer(true); return; }
    if (msg.type === "answer") return onAnswer(msg.turn);
    if (msg.type === "trace") return downloadTrace(msg.events);
    if (msg.type === "replay_done") { $("btn-play").disabled = false; window.__replayDone = true; return; }
    if (msg.type === "error") logLine({ t_stream: 0, event: "error", detail: msg.message });
  }

  function onSession(msg) {
    state.turns = {}; state.current = null; state.queries = {};
    $("transcript").innerHTML = ""; $("events").innerHTML = ""; $("queries").innerHTML = "";
    $("answer").innerHTML = '<p class="placeholder">Listening… retrieval starts as soon as an intent is stable.</p>';
    $("answer-badges").innerHTML = ""; $("diffbar").innerHTML = ""; $("source").hidden = true;
    ["ttft", "lead", "queries", "ground", "cost", "version"].forEach((k) => ($("kpi-" + k).textContent = "–"));
    $("session-pill").textContent = msg.session_id;
    const c = msg.config || {};
    $("config-chips").innerHTML = [
      `controller: ${c.controller}`, `retrieval: ${c.retrieval}${c.rerank ? " + rerank" : ""}`,
      `synthesis: ${c.synthesis}`, `${c.corpus_chunks} chunks`,
    ].map((x) => `<span class="chip">${esc(x)}</span>`).join("");
  }

  // ------------------------------------------------------------------ events
  function onEvent(ev) {
    const tid = ev.turn_id;
    switch (ev.event) {
      case "turn_started": newTurn(tid); break;
      case "chunk_received": addChunk(tid, ev); break;
      case "controller_decision": markDecision(tid, ev); break;
      case "retrieval_started": addQuery(ev); break;
      case "retrieval_completed": completeQuery(ev); break;
      case "decomposition": (ev.sub_queries || []).forEach((q) => addQuery({ ...q, reused: true, final: true })); break;
      case "retrieval_suppressed": setKind(tid, ev.reason); break;
      case "first_token": $("kpi-ttft").textContent = fmtMs(ev.ttft_ms); break;
      case "answer_version": onVersion(ev); break;
      case "grounding_check": $("kpi-ground").textContent = ev.claims ? `${ev.supported}/${ev.claims}` : "–"; break;
      case "turn_completed": onTurnDone(ev); break;
    }
    logLine(ev);
  }

  function newTurn(tid) {
    const el = document.createElement("div");
    el.className = "turn";
    el.innerHTML = `<div class="turn-head"><span>${esc(tid)}</span><span class="turn-kind">listening</span></div>
                    <div class="utter"></div>`;
    $("transcript").appendChild(el);
    state.turns[tid] = { el, utter: el.querySelector(".utter"), kind: el.querySelector(".turn-kind") };
    state.current = tid; state.livePill = null; state.lastChunkEl = null;
    state.queries = {}; $("queries").innerHTML = "";
    state.answerText = ""; state.answerIntents = [];
    ["ttft", "lead", "queries", "ground", "cost"].forEach((k) => ($("kpi-" + k).textContent = "…"));
    scrollEnd($("transcript"));
  }

  function addChunk(tid, ev) {
    const t = state.turns[tid]; if (!t) return;
    const cumulative = ev.text === ev.partial && state.livePill;
    let pill = cumulative ? state.livePill : null;
    if (!pill) {
      pill = document.createElement("span");
      pill.className = "chunk";
      pill.innerHTML = `<span class="tx"></span><span class="ts"></span><span class="dec"></span>`;
      t.utter.appendChild(pill);
      if (ev.text === ev.partial) state.livePill = pill;
    }
    pill.querySelector(".tx").textContent = ev.text;
    pill.querySelector(".ts").textContent = `${ev.t_stream.toFixed(2)}s`;
    state.lastChunkEl = pill;
    scrollEnd($("transcript"));
  }

  function markDecision(tid, ev) {
    const t = state.turns[tid]; if (!t) return;
    if (ev.final) {
      const end = document.createElement("span");
      end.className = "chunk end";
      end.innerHTML = `<span class="tx">⏹ end of utterance</span><span class="ts">${ev.t_stream.toFixed(2)}s</span>
                       <span class="dec retrieve">${esc(ev.gate.label.toUpperCase())}</span>`;
      t.utter.appendChild(end);
      setKind(tid, ev.gate.label);
      return;
    }
    const pill = state.lastChunkEl; if (!pill) return;
    const d = pill.querySelector(".dec");
    const a = ev.action;
    d.className = "dec " + a;
    d.textContent = a === "retrieve" ? "RETRIEVE" : a === "suppress" ? "SUPPRESS" : "WAIT";
    pill.title = ev.reason || "";
    if (a === "retrieve") pill.classList.add("fired");
  }

  function setKind(tid, kind) {
    const t = state.turns[tid]; if (!t) return;
    t.kind.textContent = kind; t.kind.className = "turn-kind " + kind;
  }

  function addQuery(q) {
    const id = q.qid;
    if (state.queries[id]) return;
    if (q.final) {   // a final sub-query that re-used a speculative search
      const dup = Object.values(state.queries).find((x) => x.text === q.text);
      if (dup) return;
      if (q.trigger !== "final") return;
    }
    const el = document.createElement("div");
    el.className = "q";
    const trig = q.trigger || "final";
    const label = trig === "multi_intent" ? "multi-intent" : trig;
    el.innerHTML = `<span class="trig ${esc(trig)}">${esc(label)}</span>${esc(q.query || q.text)}
                    <div class="hits">searching…</div>`;
    $("queries").appendChild(el);
    state.queries[id] = { el, text: q.query || q.text };
    if (state.lastChunkEl && trig !== "final") {
      const d = state.lastChunkEl.querySelector(".dec");
      d.textContent = "RETRIEVE · " + label;
    }
    scrollEnd($("queries"));
  }

  function completeQuery(ev) {
    (ev.qids || []).forEach((qid) => {
      const q = state.queries[qid]; if (!q) return;
      const top = (ev.top && ev.top[qid]) || [];
      q.el.querySelector(".hits").textContent =
        `${fmtMs(ev.compute_ms)} · ${ev.used === false ? "superseded · " : ""}top: ${top.join(", ")}`;
    });
  }

  function onVersion(ev) {
    $("kpi-version").textContent = "v" + ev.version;
    const b = [`<span class="badge v">v${ev.version}</span>`, `<span class="badge">${esc(ev.kind)}</span>`];
    if (ev.view && ev.view !== "full") b.push(`<span class="badge">view: ${esc(ev.view)}</span>`);
    $("answer-badges").innerHTML = b.join("");
    const d = ev.diff;
    if (d && d.mode === "refine") {
      $("diffbar").innerHTML = `Refined v${d.parent_version} → v${ev.version}: <b>${d.added.length}</b> claims added,
        <b>${d.retained.length}</b> retained, <b>${d.retired.length}</b> retired · retrieval scope: delta only
        (${(d.targets || []).length} quer${(d.targets || []).length === 1 ? "y" : "ies"})`;
    } else if (d && d.mode === "presentation") {
      $("diffbar").innerHTML = `Presentation-only turn: <b>no retrieval</b> · reformatted v${ev.version}
        as <b>${esc(d.style)}${d.count ? " × " + d.count : ""}</b> · citations retained`;
    } else if (ev.kind === "chitchat") {
      $("diffbar").innerHTML = `Conversational turn: <b>retrieval suppressed</b> · answer v${ev.version} kept in session`;
    } else if (d) {
      $("diffbar").innerHTML = `New answer v${ev.version} · <b>${(d.added || []).length}</b> grounded claims`;
    }
  }

  function onTurnDone(ev) {
    const L = ev.latency || {}, C = ev.cost || {}, S = ev.speculative || {};
    $("kpi-lead").textContent = ev.early_retrieval && L.retrieval_lead_ms != null ? fmtMs(L.retrieval_lead_ms) :
      ev.retrieved ? "at end" : "none";
    $("kpi-queries").textContent = ev.retrieved ? `${S.dispatched || 0} (${S.reused || 0})` : "0";
    $("kpi-cost").textContent = C.usd != null ? "$" + Number(C.usd).toFixed(6) : "–";
    if (L.ttft_ms == null) $("kpi-ttft").textContent = "–";
  }

  function onAnswer(turn) {
    state.answerText = turn.text || "";
    state.answerIntents = (turn.intents || []).map((i) => i.label);
    if (!state.answerText) {
      $("answer").innerHTML = `<p class="placeholder">No retrieval needed for this turn (${esc(turn.kind)}).
        The current answer (v${turn.version}) stays in session memory.</p>`;
    } else renderAnswer(false);
  }

  // ------------------------------------------------------------------ answer rendering
  const CITE = /\[((?:[A-Za-z0-9_]+ §\d+)(?:, [A-Za-z0-9_]+ §\d+)*)\]/g;
  function fmtLine(s) {
    return esc(s).replace(CITE, (_, labels) =>
      labels.split(", ").map((l) => `<span class="cite" data-label="${l}">${l}</span>`).join(""));
  }
  function renderAnswer(streaming) {
    const lines = state.answerText.split("\n");
    let html = "", inList = false;
    const labels = state.answerIntents.slice().sort((a, b) => b.length - a.length);
    for (const raw of lines) {
      const line = raw.trim(); if (!line) continue;
      if (line.startsWith("- ")) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${fmtLine(line.slice(2))}</li>`; continue;
      }
      if (inList) { html += "</ul>"; inList = false; }
      if (line.startsWith("Not verified:")) { html += `<div class="unc">${fmtLine(line.slice(13).trim())}</div>`; continue; }
      if (line.startsWith("Update")) { html += `<p class="update">${fmtLine(line)}</p>`; continue; }
      let cls = "", body = line, label = "";
      if (line.startsWith("Still applies - ")) { cls = "retained"; body = line.slice(16); label = "Still applies · "; }
      const hit = labels.find((l) => body.startsWith(l + ":"));
      if (hit) { label += hit; body = body.slice(hit.length + 1); }
      html += `<p class="${cls}">${label ? `<span class="label">${esc(label)}:</span> ` : ""}${fmtLine(body)}</p>`;
    }
    if (inList) html += "</ul>";
    if (streaming) html += '<span class="caret"></span>';
    $("answer").innerHTML = html;
    scrollEnd($("answer"));
  }

  $("answer").addEventListener("click", async (e) => {
    const el = e.target.closest(".cite"); if (!el) return;
    const r = await fetch("/api/chunk?label=" + encodeURIComponent(el.dataset.label));
    if (!r.ok) return;
    const c = await r.json();
    $("source-label").textContent = c.label;
    $("source-title").textContent = ` ${c.doc_title} · ${c.section_title}${c.status === "superseded" ?
      " · SUPERSEDED by " + c.superseded_by : ""}`;
    $("source-body").textContent = c.text;
    $("source").hidden = false;
  });
  $("source-close").onclick = () => ($("source").hidden = true);

  // ------------------------------------------------------------------ event log
  function logLine(ev) {
    if (ev.event === "chunk_received") return;       // chunks are shown in the transcript
    const el = document.createElement("div");
    el.className = "ev " + ev.event;
    el.innerHTML = `<span class="t">${(ev.t_stream || 0).toFixed(2)}s</span><span class="n">${esc(ev.event)}</span>
                    <span class="d">${esc(summary(ev))}</span>`;
    $("events").appendChild(el);
    while ($("events").children.length > 400) $("events").firstChild.remove();
    scrollEnd($("events"));
  }
  function summary(ev) {
    switch (ev.event) {
      case "controller_decision": return `${ev.action}${ev.gate ? " · " + ev.gate.label : ""} · ${ev.reason || ""}`;
      case "retrieval_started": return `[${ev.trigger}] ${ev.query}`;
      case "retrieval_completed": return `${fmtMs(ev.compute_ms)} · ${Object.values(ev.top || {}).map((x) => x[0]).join(", ")}`;
      case "decomposition": return `${(ev.sub_queries || []).length} sub-queries${ev.scope ? " (" + ev.scope + ")" : ""}`;
      case "evidence_fused": return (ev.chunks || []).slice(0, 6).join(", ");
      case "first_token": return `TTFT ${fmtMs(ev.ttft_ms)}`;
      case "answer_version": return `v${ev.version} · ${ev.kind} · ${(ev.citations || []).length} citations`;
      case "grounding_check": return `${ev.supported}/${ev.claims} claims supported · fabricated ${ev.fabricated_ids.length}`;
      case "uncertainty_flag": return ev.message || "";
      case "retrieval_suppressed": return `${ev.reason}: ${ev.detail || ""}`;
      case "turn_completed": return `${ev.kind} · $${Number((ev.cost || {}).usd || 0).toFixed(6)} · ${(ev.cost || {}).cpu_ms} ms CPU`;
      case "session_started": return JSON.stringify(ev.config);
      default: return ev.detail || "";
    }
  }

  // ------------------------------------------------------------------ inputs
  async function loadScenarios() {
    const list = await (await fetch("/api/scenarios")).json();
    $("scenario").innerHTML = list.map((s) => `<option value="${esc(s.session_id)}">${esc(s.title)}</option>`).join("");
  }
  $("btn-play").onclick = () => {
    $("btn-play").disabled = true; window.__replayDone = false;
    send({ type: "replay", scenario: $("scenario").value, speed: Number($("speed").value) });
    setTimeout(() => ($("btn-play").disabled = false), 240000);
  };
  $("btn-new").onclick = () => send({ type: "new_session" });
  $("btn-trace").onclick = () => send({ type: "trace" });

  // typed text is streamed like an ASR: 2-4 words every ~1 s (2.7 words/s)
  $("btn-send").onclick = async () => {
    const text = $("typed").value.trim(); if (!text) return;
    $("typed").value = "";
    send({ type: "start_turn" });
    const words = text.split(/\s+/);
    for (let i = 0; i < words.length;) {
      const n = 2 + Math.floor(Math.random() * 3);
      const part = words.slice(i, i + n).join(" "); i += n;
      await sleep((n / 2.7) * 1000);
      send({ type: "chunk", text: part });
    }
    await sleep(350);
    send({ type: "end_turn" });
  };
  $("typed").addEventListener("keydown", (e) => { if (e.key === "Enter") $("btn-send").click(); });

  // live microphone through the browser's streaming speech recogniser (Chrome / Edge)
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("btn-mic").disabled = true; $("btn-mic").title = "Web Speech API not available in this browser"; }
  $("btn-mic").onclick = () => {
    if (!SR) return;
    if (state.micOn) { state.rec && state.rec.stop(); return; }
    const rec = new SR();
    rec.lang = "en-IN"; rec.interimResults = true; rec.continuous = false;
    let lastSent = "";
    rec.onstart = () => { state.micOn = true; $("btn-mic").classList.add("live"); send({ type: "start_turn" }); };
    rec.onresult = (e) => {
      let txt = "";
      for (let i = 0; i < e.results.length; i++) txt += e.results[i][0].transcript;
      txt = txt.trim();
      if (txt && txt !== lastSent) { lastSent = txt; send({ type: "chunk", text: txt, cumulative: true }); }
    };
    rec.onend = () => { state.micOn = false; $("btn-mic").classList.remove("live"); send({ type: "end_turn" }); };
    state.rec = rec; rec.start();
  };

  function downloadTrace(events) {
    const blob = new Blob([events.map((e) => JSON.stringify(e)).join("\n") + "\n"], { type: "application/jsonl" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `duplexrag-trace-${Date.now()}.jsonl`;
    a.click();
  }

  // ------------------------------------------------------------------ utils
  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function fmtMs(v) { if (v == null) return "–"; v = Number(v); return v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms"; }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
  function scrollEnd(el) { el.scrollTop = el.scrollHeight; }

  loadScenarios();
  connect();
})();
