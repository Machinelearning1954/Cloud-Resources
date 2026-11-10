"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids);
  return node;
};

const DIRECTIONS = [
  "slow dolly in", "crane shot rising", "handheld", "aerial orbit", "rack focus",
  "golden hour light", "high-contrast noir lighting", "anamorphic wide shot", "shallow depth of field",
];

const state = { config: null, kind: "video", model: null, jobs: new Map(), directions: new Set() };

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function fillSelect(select, options, keep) {
  const previous = keep ? select.value : null;
  select.replaceChildren(...options.map(([value, label]) => el("option", { value, textContent: label })));
  if (previous && options.some(([v]) => String(v) === previous)) select.value = previous;
}

function currentModel() {
  return state.config.models.find((m) => m.id === state.model);
}

function renderControls() {
  const models = state.config.models.filter((m) => m.kind === state.kind);
  if (!models.some((m) => m.id === state.model)) state.model = models[0].id;

  document.querySelectorAll(".seg button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.kind === state.kind)));

  $("models").replaceChildren(...models.map((m) => {
    const card = el("button", { type: "button", className: "model" },
      el("b", {}, m.name, el("span", { textContent: `${m.licence} · free` })),
      el("em", { textContent: m.blurb }));
    card.setAttribute("aria-pressed", String(m.id === state.model));
    card.onclick = () => { state.model = m.id; renderControls(); };
    return card;
  }));

  const model = currentModel();
  fillSelect($("aspect"), state.config.aspects[state.kind].map((a) => [a, a]), true);

  const isVideo = state.kind === "video";
  $("secondsField").hidden = !isVideo;
  $("countField").hidden = isVideo;
  if (isVideo) {
    const lengths = [];
    for (let s = 2; s <= model.max_seconds; s += s < 6 ? 1 : 2) lengths.push([s, `${s} s`]);
    fillSelect($("seconds"), lengths, true);
    if (!$("seconds").dataset.touched) $("seconds").value = String(Math.min(5, model.max_seconds));
  }
  $("negativeField").hidden = !model.supports_negative;
}

function renderChips() {
  $("chips").replaceChildren(...DIRECTIONS.map((text) => {
    const chip = el("button", { type: "button", className: "chip", textContent: text });
    chip.setAttribute("aria-pressed", "false");
    chip.onclick = () => {
      const on = !state.directions.has(text);
      on ? state.directions.add(text) : state.directions.delete(text);
      chip.setAttribute("aria-pressed", String(on));
    };
    return chip;
  }));
}

function takeCard(job, number) {
  const stage = el("div", { className: "stage" });
  stage.style.aspectRatio = `${job.width} / ${job.height}`;
  const src = job.file ? `/media/${encodeURIComponent(job.file)}` : null;

  if (job.status === "done" && job.media === "video") {
    stage.append(el("video", { src, controls: true, loop: true, playsInline: true, preload: "metadata" }));
  } else if (job.status === "done") {
    stage.append(el("img", { src, alt: job.prompt, loading: "lazy" }));
  } else if (job.status === "error") {
    stage.append(el("div", { className: "status bad", textContent: job.error || "This take failed." }));
  } else {
    const label = job.status === "queued" ? "Waiting for the camera…" : `Rolling… ${Math.round(job.progress * 100)}%`;
    const bar = el("div", { className: "bar" });
    bar.style.width = `${job.progress * 100}%`;
    stage.append(el("div", { className: "status", textContent: label }), bar);
  }

  const info = [`Take ${number}`, job.model_name, job.aspect];
  if (job.seconds) info.push(`${job.seconds} s`);
  info.push(`seed ${job.seed}`);
  if (job.elapsed) info.push(`${job.elapsed} s render`);

  const tools = el("div", { className: "tools" });
  if (src && job.status === "done") tools.append(el("a", { href: src, download: job.file, textContent: "Download" }));
  tools.append(el("button", { type: "button", textContent: "Reuse scene", onclick: () => {
    $("prompt").value = job.prompt;
    $("seed").value = job.seed;
    state.kind = job.kind; state.model = job.model;
    renderControls();
    $("prompt").focus();
  } }));
  if (job.status !== "running") tools.append(el("button", { type: "button", textContent: "Delete", onclick: async () => {
    try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); } catch (e) { return showError(e.message); }
    state.jobs.delete(job.id);
    renderTakes();
  } }));

  return el("article", { className: "take" }, stage,
    el("div", { className: "meta" },
      el("p", { textContent: job.prompt }),
      el("div", { className: "info", textContent: info.join(" · ") }),
      tools));
}

let lastSignature = "";
function renderTakes() {
  const jobs = [...state.jobs.values()].sort((a, b) => b.created - a.created);
  // Skip the rebuild when nothing changed, so playing videos are not reset by polling.
  const signature = jobs.map((j) => `${j.id}:${j.status}:${j.progress}`).join("|");
  if (signature === lastSignature) return;
  lastSignature = signature;
  $("takes").replaceChildren(...jobs.map((job, i) => takeCard(job, jobs.length - i)));
  $("empty").hidden = jobs.length > 0;
  $("count-label").textContent = jobs.length ? `${jobs.length} take${jobs.length === 1 ? "" : "s"}` : "";
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}

async function poll() {
  const active = [...state.jobs.values()].some((j) => j.status === "queued" || j.status === "running");
  if (active) {
    try {
      const { jobs } = await api("/api/jobs");
      state.jobs = new Map(jobs.map((j) => [j.id, j]));
      renderTakes();
    } catch { /* server briefly unreachable; try again on the next tick */ }
  }
  setTimeout(poll, 1000);
}

$("slate").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  const scene = $("prompt").value.trim();
  if (!scene) return showError("Write a scene first.");
  const body = {
    model: state.model,
    prompt: [scene, ...state.directions].join(", "),
    aspect: $("aspect").value,
    negative_prompt: currentModel().supports_negative ? $("negative").value : "",
    seed: $("seed").value,
    seconds: $("seconds").value,
    count: state.kind === "image" ? $("count").value : 1,
  };
  $("action").disabled = true;
  try {
    const { jobs } = await api("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    jobs.forEach((j) => state.jobs.set(j.id, j));
    renderTakes();
  } catch (e) {
    showError(e.message);
  } finally {
    $("action").disabled = false;
  }
});

document.querySelectorAll(".seg button").forEach((b) => {
  b.onclick = () => { state.kind = b.dataset.kind; renderControls(); };
});
$("seconds").addEventListener("change", () => { $("seconds").dataset.touched = "1"; });

(async function start() {
  try {
    state.config = await api("/api/config");
    const { jobs } = await api("/api/jobs");
    state.jobs = new Map(jobs.map((j) => [j.id, j]));
  } catch (e) {
    return showError(`Could not reach the studio server: ${e.message}`);
  }
  $("engine").textContent = state.config.engine === "preview" ? "Preview engine" : "Local GPU engine";
  if (state.config.engine === "preview") {
    $("notice").textContent = state.config.engine_note;
    $("notice").hidden = false;
  }
  fillSelect($("count"), Array.from({ length: state.config.max_batch }, (_, i) => [i + 1, String(i + 1)]));
  renderChips();
  renderControls();
  renderTakes();
  poll();
})();
