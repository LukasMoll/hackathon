const PLAYER_COLORS = ["#f5ea14", "#19d9e5", "#e600ff", "#ff1f1f", "#1aff1a", "#1f2bff"];
const PLAYBACK_INTERVAL_MS = 1000;

const state = {
  bots: new Array(6).fill(""),
  activeBot: 0,
  boardRadius: 5,
  boardTiles: [],
  frames: [],
  logs: {},
  errors: {},
  winnerId: null,
  playTimer: null,
};

const elements = {
  botList: document.getElementById("bot-list"),
  activeBotTitle: document.getElementById("active-bot-title"),
  codeEditor: document.getElementById("code-editor"),
  fileInput: document.getElementById("file-input"),
  loadTemplateBtn: document.getElementById("load-template-btn"),
  runBtn: document.getElementById("run-btn"),
  seedInput: document.getElementById("seed-input"),
  stepsInput: document.getElementById("steps-input"),
  canvas: document.getElementById("game-canvas"),
  playBtn: document.getElementById("play-btn"),
  frameSlider: document.getElementById("frame-slider"),
  frameLabel: document.getElementById("frame-label"),
  statusLine: document.getElementById("status-line"),
  logOutput: document.getElementById("log-output"),
};

const ctx = elements.canvas.getContext("2d");

async function boot() {
  createBotButtons();
  wireEvents();

  const [templateResp, boardResp] = await Promise.all([
    fetch("/api/default-bot").then((r) => r.json()),
    fetch("/api/board").then((r) => r.json()),
  ]);

  state.boardRadius = boardResp.radius;
  state.boardTiles = boardResp.tiles;

  for (let i = 0; i < 6; i += 1) {
    state.bots[i] = templateResp.code;
  }

  setActiveBot(0);
  drawEmptyBoard();
}

function createBotButtons() {
  elements.botList.innerHTML = "";
  for (let i = 0; i < 6; i += 1) {
    const btn = document.createElement("button");
    btn.className = "bot-btn";
    btn.textContent = `P${i + 1}`;
    btn.style.background = PLAYER_COLORS[i];
    btn.style.color = "#000";
    btn.dataset.playerId = String(i);
    btn.addEventListener("click", () => setActiveBot(i));
    elements.botList.appendChild(btn);
  }
}

function wireEvents() {
  elements.codeEditor.addEventListener("input", () => {
    state.bots[state.activeBot] = elements.codeEditor.value;
  });

  elements.fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    state.bots[state.activeBot] = text;
    elements.codeEditor.value = text;
    elements.fileInput.value = "";
  });

  elements.loadTemplateBtn.addEventListener("click", async () => {
    const templateResp = await fetch("/api/default-bot").then((r) => r.json());
    state.bots[state.activeBot] = templateResp.code;
    elements.codeEditor.value = templateResp.code;
  });

  elements.runBtn.addEventListener("click", runSimulation);

  elements.playBtn.addEventListener("click", () => {
    if (state.playTimer) {
      stopPlayback();
      return;
    }
    startPlayback();
  });

  elements.frameSlider.addEventListener("input", () => {
    const frameIndex = Number(elements.frameSlider.value);
    drawFrame(frameIndex);
  });

  window.addEventListener("resize", () => {
    if (!state.frames.length) {
      drawEmptyBoard();
      return;
    }
    drawFrame(Number(elements.frameSlider.value));
  });
}

function setActiveBot(index) {
  state.activeBot = index;
  elements.activeBotTitle.textContent = `Bot ${index + 1}`;
  elements.codeEditor.value = state.bots[index] ?? "";

  const buttons = elements.botList.querySelectorAll(".bot-btn");
  buttons.forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.playerId) === index);
  });
}

async function runSimulation() {
  stopPlayback();
  state.bots[state.activeBot] = elements.codeEditor.value;

  elements.statusLine.textContent = "Running simulation...";

  const payload = {
    bots: state.bots,
    seed: Number(elements.seedInput.value || 1),
    max_steps: Number(elements.stepsInput.value || 220),
  };

  const resp = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const text = await resp.text();
    elements.statusLine.textContent = `Simulation failed: ${text}`;
    return;
  }

  const data = await resp.json();
  state.frames = data.frames;
  state.logs = data.logs;
  state.errors = data.errors;
  state.winnerId = data.winner_id;
  state.boardRadius = data.board_radius;

  elements.frameSlider.min = "0";
  elements.frameSlider.max = String(Math.max(0, state.frames.length - 1));
  elements.frameSlider.value = "0";

  drawFrame(0);
  renderLogs();

  const winnerLabel = data.winner_id === null ? "No winner (max steps reached)" : `Winner: Player ${data.winner_id + 1}`;
  elements.statusLine.textContent = `${winnerLabel} | Steps: ${data.steps_played} | Ticks: ${data.total_ticks}`;
}

function startPlayback() {
  if (!state.frames.length) return;
  elements.playBtn.textContent = "Pause";
  state.playTimer = setInterval(() => {
    const cur = Number(elements.frameSlider.value);
    const max = Number(elements.frameSlider.max);
    if (cur >= max) {
      stopPlayback();
      return;
    }
    elements.frameSlider.value = String(cur + 1);
    drawFrame(cur + 1);
  }, PLAYBACK_INTERVAL_MS);
}

function stopPlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  elements.playBtn.textContent = "Play";
}

function renderLogs() {
  const lines = [];
  for (let i = 0; i < 6; i += 1) {
    lines.push(`[Player ${i + 1}]`);
    const botLogs = state.logs[i] || [];
    if (!botLogs.length) {
      lines.push("(no output)");
    } else {
      lines.push(...botLogs);
    }
    if (state.errors[i]) {
      lines.push(`ERROR: ${state.errors[i]}`);
    }
    lines.push("");
  }
  elements.logOutput.textContent = lines.join("\n");
}

function drawEmptyBoard() {
  drawFrameInternal({ players: [], gems: [], step: 0 }, true);
  elements.frameLabel.textContent = "Frame 0 / 0";
}

function drawFrame(index) {
  if (!state.frames.length) {
    drawEmptyBoard();
    return;
  }
  const frame = state.frames[index];
  drawFrameInternal(frame, false);
  elements.frameLabel.textContent = `Frame ${index} / ${state.frames.length - 1}`;
}

function drawFrameInternal(frame, noPlayers) {
  fitCanvasToContainer();
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  const radius = state.boardRadius;
  const hexSize = computeHexSize(radius);
  const centerX = elements.canvas.width / 2;
  const centerY = elements.canvas.height / 2;

  for (const [q, r] of state.boardTiles) {
    const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
    const isCenter = q === 0 && r === 0;
    drawHex(x, y, hexSize, isCenter ? "#e7d99b" : "#cfc086", "#51482f");
  }

  for (const [q, r] of frame.gems || []) {
    const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
    ctx.fillStyle = "#2be66f";
    ctx.beginPath();
    ctx.arc(x, y, hexSize * 0.26, 0, Math.PI * 2);
    ctx.fill();
  }

  if (!noPlayers) {
    for (const p of frame.players) {
      for (const [q, r] of p.body) {
        const { x, y } = axialToPixel(q, r, hexSize, centerX, centerY);
        ctx.fillStyle = p.alive ? p.color : "#666";
        ctx.globalAlpha = p.alive ? 1.0 : 0.7;
        ctx.beginPath();
        ctx.arc(x, y, hexSize * 0.33, 0, Math.PI * 2);
        ctx.fill();
      }

      const { x, y } = axialToPixel(p.q, p.r, hexSize, centerX, centerY);
      ctx.globalAlpha = p.alive ? 1.0 : 0.8;
      ctx.fillStyle = p.alive ? p.color : "#6d6d6d";
      ctx.beginPath();
      ctx.arc(x, y, hexSize * 0.45, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#101010";
      ctx.lineWidth = 2;
      ctx.stroke();

      drawEyes(x, y, hexSize, p.facing);
    }
  }

  ctx.globalAlpha = 1;
}

function fitCanvasToContainer() {
  const rect = elements.canvas.getBoundingClientRect();
  const width = Math.max(420, Math.floor(rect.width));
  const height = Math.max(420, Math.floor(window.innerHeight * 0.62));

  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
}

function computeHexSize(radius) {
  const w = elements.canvas.width;
  const h = elements.canvas.height;

  const sizeByWidth = w / (Math.sqrt(3) * (2 * radius + 2));
  const sizeByHeight = h / (1.5 * (2 * radius + 1) + 2);
  return Math.max(12, Math.min(sizeByWidth, sizeByHeight));
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

function drawEyes(x, y, size, facing) {
  const pupilOffset = directionToVector(facing, size * 0.08);
  const eyeOffset = size * 0.14;

  drawEye(x - eyeOffset, y - eyeOffset * 0.25, size * 0.11, pupilOffset);
  drawEye(x + eyeOffset, y - eyeOffset * 0.25, size * 0.11, pupilOffset);
}

function drawEye(x, y, radius, pupilOffset) {
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#0a0a0a";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "#0a0a0a";
  ctx.beginPath();
  ctx.arc(x + pupilOffset.x, y + pupilOffset.y, radius * 0.45, 0, Math.PI * 2);
  ctx.fill();
}

function directionToVector(facing, length) {
  const dirs = [
    { x: 0, y: -1 },
    { x: 0.9, y: -0.5 },
    { x: 0.9, y: 0.5 },
    { x: 0, y: 1 },
    { x: -0.9, y: 0.5 },
    { x: -0.9, y: -0.5 },
  ];
  const d = dirs[((facing % 6) + 6) % 6];
  return { x: d.x * length, y: d.y * length };
}

boot().catch((err) => {
  elements.statusLine.textContent = `Initialization failed: ${err.message}`;
});
