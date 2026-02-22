const PLAYER_COLORS = ["#f5ea14", "#19d9e5", "#e600ff", "#ff1f1f", "#1aff1a", "#1f2bff"];

const state = {
  sessionId: null,
  snapshot: null,
  frameIndex: 0,
  autoplay: null,
};

const el = {
  seed: document.getElementById("seed"),
  maxSteps: document.getElementById("max-steps"),
  newSession: document.getElementById("new-session"),
  sessionInfo: document.getElementById("session-info"),
  actions: document.getElementById("actions"),
  stepOnce: document.getElementById("step-once"),
  stepRandom: document.getElementById("step-random"),
  autoPlay: document.getElementById("auto-play"),
  saveName: document.getElementById("save-name"),
  saveRun: document.getElementById("save-run"),
  savedList: document.getElementById("saved-list"),
  refreshSaved: document.getElementById("refresh-saved"),
  loadSaved: document.getElementById("load-saved"),
  queryPlayer: document.getElementById("query-player"),
  queryFn: document.getElementById("query-fn"),
  queryArgs: document.getElementById("query-args"),
  queryBtn: document.getElementById("query-btn"),
  queryOut: document.getElementById("query-out"),
  canvas: document.getElementById("ml-canvas"),
  play: document.getElementById("play"),
  frame: document.getElementById("frame"),
  frameLabel: document.getElementById("frame-label"),
  status: document.getElementById("status"),
  log: document.getElementById("ml-log"),
};

const ctx = el.canvas.getContext("2d");

function actionValue(i) {
  const sel = document.getElementById(`act-${i}`);
  return sel ? sel.value : "forward";
}

function buildActionControls() {
  el.actions.innerHTML = "";
  for (let i = 0; i < 6; i += 1) {
    const label = document.createElement("div");
    label.textContent = `Player ${i + 1}`;
    label.style.color = PLAYER_COLORS[i];
    label.style.fontWeight = "700";

    const sel = document.createElement("select");
    sel.id = `act-${i}`;
    sel.innerHTML = `
      <option value="left">Left</option>
      <option value="forward" selected>Forward</option>
      <option value="right">Right</option>
    `;

    el.actions.appendChild(label);
    el.actions.appendChild(sel);
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

function setSnapshot(snapshot) {
  state.snapshot = snapshot;
  const frameCount = snapshot.frames.length;
  el.frame.min = "0";
  el.frame.max = String(Math.max(0, frameCount - 1));
  state.frameIndex = Math.max(0, frameCount - 1);
  el.frame.value = String(state.frameIndex);
  renderFrame(state.frameIndex);
  el.status.textContent = `Step ${snapshot.step}/${snapshot.max_steps} | done=${snapshot.done} | winner=${snapshot.winner_id}`;
}

async function newSession() {
  const payload = {
    seed: Number(el.seed.value || 1),
    max_steps: Number(el.maxSteps.value || 300),
  };
  const data = await api("/api/ml/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.sessionId = data.session_id;
  el.sessionInfo.textContent = `Session ${data.session_id} (created ${data.created_at})`;
  setSnapshot(data.snapshot);
}

async function stepOnce(randomize = false) {
  if (!state.sessionId) return;
  const actions = [];
  for (let i = 0; i < 6; i += 1) {
    if (randomize) {
      const options = ["left", "forward", "right"];
      const v = options[Math.floor(Math.random() * options.length)];
      document.getElementById(`act-${i}`).value = v;
      actions.push(v);
    } else {
      actions.push(actionValue(i));
    }
  }

  const data = await api("/api/ml/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, actions }),
  });
  setSnapshot(data.snapshot);
}

async function refreshSaved() {
  const data = await api("/api/ml/saved");
  el.savedList.innerHTML = "";
  for (const file of data.files) {
    const opt = document.createElement("option");
    opt.value = file;
    opt.textContent = file;
    el.savedList.appendChild(opt);
  }
}

async function saveRun() {
  if (!state.sessionId) return;
  const data = await api("/api/ml/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, name: el.saveName.value || "ml_run" }),
  });
  await refreshSaved();
  el.status.textContent = `Saved ${data.filename}`;
}

async function loadSaved() {
  const filename = el.savedList.value;
  if (!filename) return;
  const data = await api("/api/ml/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  state.sessionId = data.session_id;
  setSnapshot(data.snapshot);
  el.sessionInfo.textContent = `Loaded replay ${filename}`;
}

async function callFn() {
  if (!state.sessionId) return;
  const args = (el.queryArgs.value || "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s));

  const data = await api("/api/ml/call", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      player_id: Number(el.queryPlayer.value || 0),
      function: el.queryFn.value,
      args,
    }),
  });

  el.queryOut.textContent = JSON.stringify(data.value);
}

function axialToPixel(q, r, size, cx, cy) {
  const x = size * Math.sqrt(3) * (q + r / 2);
  const y = size * 1.5 * r;
  return { x: cx + x, y: cy + y };
}

function drawHex(cx, cy, size, fill, stroke) {
  ctx.beginPath();
  for (let i = 0; i < 6; i += 1) {
    const angle = ((60 * i - 30) * Math.PI) / 180;
    const x = cx + size * Math.cos(angle);
    const y = cy + size * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

function renderFrame(index) {
  if (!state.snapshot) return;
  const frame = state.snapshot.frames[index];
  if (!frame) return;

  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);

  const centerX = el.canvas.width / 2;
  const centerY = el.canvas.height / 2;
  const hexSize = 30;

  for (let q = -5; q <= 5; q += 1) {
    for (let r = -5; r <= 5; r += 1) {
      const s = -q - r;
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > 5) continue;
      const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
      const center = q === 0 && r === 0;
      drawHex(x, y, hexSize, center ? "#e7d99b" : "#cfc086", "#51482f");
    }
  }

  for (const [q, r] of frame.gems) {
    const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
    ctx.fillStyle = "#2be66f";
    ctx.beginPath();
    ctx.arc(x, y, hexSize * 0.26, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const p of frame.players) {
    for (const [q, r] of p.body) {
      const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
      ctx.fillStyle = p.alive ? p.color : "#666";
      ctx.globalAlpha = p.alive ? 1 : 0.7;
      ctx.beginPath();
      ctx.arc(x, y, hexSize * 0.33, 0, Math.PI * 2);
      ctx.fill();
    }

    const { x, y } = axialToPixel(p.q, p.r, hexSize, centerX, centerY);
    ctx.globalAlpha = p.alive ? 1 : 0.8;
    ctx.fillStyle = p.alive ? p.color : "#6d6d6d";
    ctx.beginPath();
    ctx.arc(x, y, hexSize * 0.45, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#101010";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  el.frameLabel.textContent = `Frame ${index} / ${state.snapshot.frames.length - 1}`;
  el.log.textContent = frame.events.join("\n") || "(no events)";
}

function toggleAuto() {
  if (state.autoplay) {
    clearInterval(state.autoplay);
    state.autoplay = null;
    el.autoPlay.textContent = "Auto";
    return;
  }

  state.autoplay = setInterval(async () => {
    try {
      await stepOnce(false);
      if (state.snapshot?.done) toggleAuto();
    } catch (err) {
      toggleAuto();
    }
  }, 250);
  el.autoPlay.textContent = "Stop";
}

function toggleReplayPlay() {
  if (state.autoplay) return;

  if (el.play.dataset.on === "1") {
    el.play.dataset.on = "0";
    el.play.textContent = "Play";
    if (state._replayTimer) clearInterval(state._replayTimer);
    return;
  }

  el.play.dataset.on = "1";
  el.play.textContent = "Pause";
  state._replayTimer = setInterval(() => {
    if (!state.snapshot) return;
    let i = Number(el.frame.value);
    const max = Number(el.frame.max);
    if (i >= max) {
      i = 0;
    } else {
      i += 1;
    }
    el.frame.value = String(i);
    renderFrame(i);
  }, 500);
}

function wire() {
  buildActionControls();

  el.newSession.addEventListener("click", () => newSession().catch((e) => (el.status.textContent = e.message)));
  el.stepOnce.addEventListener("click", () => stepOnce(false).catch((e) => (el.status.textContent = e.message)));
  el.stepRandom.addEventListener("click", () => stepOnce(true).catch((e) => (el.status.textContent = e.message)));
  el.autoPlay.addEventListener("click", toggleAuto);
  el.refreshSaved.addEventListener("click", () => refreshSaved().catch((e) => (el.status.textContent = e.message)));
  el.saveRun.addEventListener("click", () => saveRun().catch((e) => (el.status.textContent = e.message)));
  el.loadSaved.addEventListener("click", () => loadSaved().catch((e) => (el.status.textContent = e.message)));
  el.queryBtn.addEventListener("click", () => callFn().catch((e) => (el.queryOut.textContent = e.message)));

  el.play.addEventListener("click", toggleReplayPlay);
  el.frame.addEventListener("input", () => renderFrame(Number(el.frame.value)));

  refreshSaved().catch(() => {});
  newSession().catch((e) => (el.status.textContent = e.message));
}

wire();
