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
const LOOKS = [
  ["", "None"],
  ["shot on 35mm film, natural grain", "35mm film"],
  ["black-and-white film noir", "Film noir"],
  ["documentary realism, natural light", "Documentary"],
  ["hand-drawn animation", "Hand-drawn animation"],
  ["neon-lit science fiction", "Neon sci-fi"],
  ["sun-drenched Caribbean colour", "Caribbean sun"],
];

const state = {
  config: null, kind: "video", model: null, jobs: new Map(),
  directions: new Set(), start: null, show: "all",
};

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const mediaUrl = (job) => `/media/${encodeURIComponent(job.file)}`;

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

  // Start frame: only for video models that can animate a picture.
  const canStart = isVideo && model.supports_start_image;
  if (state.start && !state.jobs.has(state.start)) state.start = null;
  $("startField").hidden = !canStart;
  const start = canStart && state.start ? state.jobs.get(state.start) : null;
  $("startThumb").hidden = !start;
  $("startClear").hidden = !start;
  if (start) $("startThumb").src = mediaUrl(start);
  $("startHint").textContent = start ? "This picture will be the first frame." : "Press “Animate” on an image take, or";
  $("prompt").placeholder = start
    ? "Describe what happens next: the camera pushes in as the lantern flickers and rain begins"
    : "A lighthouse keeper climbs the spiral stairs at dusk, lantern swinging, storm clouds rolling in over the sea";
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

function animate(job) {
  state.kind = "video";
  state.start = job.id;
  renderControls();
  $("slate").scrollIntoView({ behavior: "smooth", block: "start" });
  $("prompt").focus({ preventScroll: true });
}

function takeCard(job, number) {
  const stage = el("div", { className: "stage" });
  stage.style.aspectRatio = `${job.width} / ${job.height}`;
  const src = job.file ? mediaUrl(job) : null;
  const isStill = job.status === "done" && job.media === "image";

  if (job.status === "done" && job.media === "video") {
    stage.append(el("video", { src, controls: true, loop: true, playsInline: true, preload: "metadata" }));
  } else if (job.status === "done") {
    const img = el("img", { src, alt: job.prompt, loading: "lazy", className: "zoomable", title: "View full size" });
    img.onclick = () => { $("viewerImg").src = src; $("viewerImg").alt = job.prompt; $("viewer").showModal(); };
    stage.append(img);
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
  if (job.start_take) info.push("from start frame");
  if (job.seed !== undefined) info.push(`seed ${job.seed}`);
  if (job.elapsed) info.push(`${job.elapsed} s render`);

  const tools = el("div", { className: "tools" });
  if (isStill) tools.append(el("button", { type: "button", textContent: "Animate", onclick: () => animate(job) }));
  if (src && job.status === "done") tools.append(el("a", { href: src, download: job.file, textContent: "Download" }));
  if (!job.uploaded) tools.append(el("button", { type: "button", textContent: "Reuse scene", onclick: () => {
    $("prompt").value = job.prompt;
    $("seed").value = job.seed;
    state.kind = job.kind; state.model = job.model;
    state.start = job.start_take && state.jobs.has(job.start_take) ? job.start_take : null;
    renderControls();
    $("prompt").focus();
  } }));
  if (job.status !== "running") tools.append(el("button", {
    type: "button", textContent: job.status === "queued" ? "Cancel" : "Delete", onclick: async () => {
      try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); } catch (e) { return showError(e.message); }
      state.jobs.delete(job.id);
      renderControls();
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
  const signature = state.show + "#" + jobs.map((j) => `${j.id}:${j.status}:${j.progress}`).join("|");
  if (signature === lastSignature) return;
  lastSignature = signature;
  const cards = [];
  jobs.forEach((job, i) => {
    if (state.show === "all" || job.kind === state.show) cards.push(takeCard(job, jobs.length - i));
  });
  $("takes").replaceChildren(...cards);
  $("empty").hidden = cards.length > 0;
  $("empty").textContent = jobs.length
    ? `No ${state.show} takes yet.`
    : "No takes yet. Describe a scene and call “Action!”";
  $("filter").hidden = jobs.length === 0;
  const waiting = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  $("count-label").textContent = jobs.length
    ? `${jobs.length} take${jobs.length === 1 ? "" : "s"}${waiting ? ` · ${waiting} in the queue` : ""}`
    : "";
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
  const model = currentModel();
  const body = {
    model: state.model,
    prompt: [scene, $("look").value, ...state.directions].filter(Boolean).join(", "),
    aspect: $("aspect").value,
    negative_prompt: model.supports_negative ? $("negative").value : "",
    seed: $("seed").value,
    seconds: $("seconds").value,
    count: state.kind === "image" ? $("count").value : 1,
    start_take: state.kind === "video" && model.supports_start_image ? state.start : null,
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

$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("slate").requestSubmit();
});

$("startUpload").addEventListener("change", async () => {
  const file = $("startUpload").files[0];
  $("startUpload").value = "";
  if (!file) return;
  showError("");
  const form = new FormData();
  form.append("image", file);
  try {
    const job = await api("/api/upload", { method: "POST", body: form });
    state.jobs.set(job.id, job);
    state.start = job.id;
    renderControls();
    renderTakes();
  } catch (e) {
    showError(e.message);
  }
});
$("startClear").onclick = () => { state.start = null; renderControls(); };

document.querySelectorAll(".seg button").forEach((b) => {
  b.onclick = () => { state.kind = b.dataset.kind; renderControls(); };
});
document.querySelectorAll("#filter button").forEach((b) => {
  b.onclick = () => {
    state.show = b.dataset.show;
    document.querySelectorAll("#filter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderTakes();
  };
});
$("seconds").addEventListener("change", () => { $("seconds").dataset.touched = "1"; });
$("viewer").addEventListener("click", (event) => { if (event.target === $("viewer")) $("viewer").close(); });

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
  fillSelect($("look"), LOOKS);
  renderChips();
  renderControls();
  renderTakes();
  poll();
})();
