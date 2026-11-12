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
const WORDING = {
  video: ["Scene", "A lighthouse keeper climbs the spiral stairs at dusk, lantern swinging, storm clouds rolling in over the sea"],
  image: ["Scene", "A lighthouse keeper climbs the spiral stairs at dusk, lantern swinging, storm clouds rolling in over the sea"],
  music: ["Music", "Warm roots reggae groove, offbeat guitar skank, deep melodic bass, steady one-drop drums"],
  voice: ["Script", "In a town by the sea, one lighthouse keeper refused to let the light go out."],
};
const VISUAL = new Set(["video", "image"]);

const state = {
  config: null, kind: "video", model: null, jobs: new Map(),
  directions: new Set(), start: null, show: "all",
  cut: { clips: [], voice: null, music: null, volume: 0.35 },
};

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
const postJson = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

const mediaUrl = (job) => `/media/${encodeURIComponent(job.file)}`;
const isDone = (id) => state.jobs.get(id)?.status === "done";

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
  const kind = state.kind;
  const visual = VISUAL.has(kind);

  $("aspectField").hidden = !visual;
  $("dirField").hidden = !visual;
  $("lookField").hidden = !visual;
  $("secondsField").hidden = kind !== "video";
  $("countField").hidden = kind !== "image";
  $("lyricsField").hidden = kind !== "music";
  $("musicSecondsField").hidden = kind !== "music";
  $("voiceField").hidden = kind !== "voice";
  $("speedField").hidden = kind !== "voice";
  $("more").hidden = kind === "voice";
  $("negativeField").hidden = !model.supports_negative;

  if (visual) fillSelect($("aspect"), state.config.aspects[kind].map((a) => [a, a]), true);
  if (kind === "video") {
    const lengths = [];
    for (let s = 2; s <= model.max_seconds; s += s < 6 ? 1 : 2) lengths.push([s, `${s} s`]);
    fillSelect($("seconds"), lengths, true);
    if (!$("seconds").dataset.touched) $("seconds").value = String(Math.min(5, model.max_seconds));
  }
  if (kind === "voice") fillSelect($("voice"), model.voices, true);

  // Start frame: only for video models that can animate a picture.
  const canStart = kind === "video" && model.supports_start_image;
  if (state.start && !state.jobs.has(state.start)) state.start = null;
  $("startField").hidden = !canStart;
  const start = canStart && state.start ? state.jobs.get(state.start) : null;
  $("startThumb").hidden = !start;
  $("startClear").hidden = !start;
  if (start) $("startThumb").src = mediaUrl(start);
  $("startHint").textContent = start ? "This picture will be the first frame." : "Press “Animate” on an image take, or";

  const [label, placeholder] = WORDING[kind];
  $("promptLabel").textContent = label;
  $("prompt").maxLength = kind === "voice" ? 5000 : 2000;
  $("prompt").placeholder = start
    ? "Describe what happens next: the camera pushes in as the lantern flickers and rain begins"
    : placeholder;
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

/* ---------- Movie cut ---------- */

function takeNumbers() {
  const ordered = [...state.jobs.values()].sort((a, b) => a.created - b.created);
  return new Map(ordered.map((job, i) => [job.id, i + 1]));
}

function renderCut() {
  const cut = state.cut;
  cut.clips = cut.clips.filter(isDone);
  if (cut.voice && !isDone(cut.voice)) cut.voice = null;
  if (cut.music && !isDone(cut.music)) cut.music = null;
  const numbers = takeNumbers();
  const anything = cut.clips.length || cut.voice || cut.music;
  $("cut").hidden = !anything;
  if (!anything) return;

  $("cutHint").textContent = cut.clips.length
    ? `${cut.clips.length} take${cut.clips.length === 1 ? "" : "s"}, played in this order`
    : "Add picture or video takes with “Add to movie”";

  $("cutClips").replaceChildren(...cut.clips.map((id, i) => {
    const job = state.jobs.get(id);
    const thumb = job.media === "video"
      ? el("div", { className: "clip-video", textContent: "▶" })
      : el("img", { src: mediaUrl(job), alt: "" });
    const controls = el("div", { className: "ctl" });
    if (i > 0) controls.append(el("button", { type: "button", className: "link", textContent: "Earlier", onclick: () => {
      [cut.clips[i - 1], cut.clips[i]] = [cut.clips[i], cut.clips[i - 1]];
      renderCut();
    } }));
    controls.append(el("button", { type: "button", className: "link", textContent: "Remove", onclick: () => {
      cut.clips.splice(i, 1);
      renderCut();
    } }));
    return el("li", {}, thumb, `Take ${numbers.get(id)}`, controls);
  }));

  const tracks = [];
  for (const [key, label] of [["voice", "Voiceover"], ["music", "Soundtrack"]]) {
    if (!cut[key]) continue;
    const row = el("div", {}, el("b", { textContent: `${label}:` }), el("span", { textContent: `Take ${numbers.get(cut[key])}` }));
    if (key === "music") {
      const slider = el("input", { type: "range", min: "0", max: "1", step: "0.05", value: String(cut.volume) });
      slider.setAttribute("aria-label", "Soundtrack volume");
      slider.oninput = () => { cut.volume = Number(slider.value); };
      row.append(el("span", { textContent: "volume" }), slider);
    }
    row.append(el("button", { type: "button", className: "link", textContent: "Remove", onclick: () => {
      cut[key] = null;
      renderCut();
    } }));
    tracks.push(row);
  }
  $("cutTracks").replaceChildren(...tracks);
  $("cutGo").disabled = cut.clips.length === 0;
}

$("cutGo").onclick = async () => {
  showError("");
  const cut = state.cut;
  $("cutGo").disabled = true;
  try {
    const job = await postJson("/api/movie", {
      clips: cut.clips, voice: cut.voice, music: cut.music,
      music_volume: cut.volume, title: $("cutTitle").value,
    });
    state.jobs.set(job.id, job);
    state.cut = { clips: [], voice: null, music: null, volume: cut.volume };
    $("cutTitle").value = "";
    renderCut();
    renderTakes();
  } catch (e) {
    showError(e.message);
    $("cut").scrollIntoView({ behavior: "smooth", block: "nearest" });
    $("cutGo").disabled = false;
  }
};

/* ---------- Reel ---------- */

function takeCard(job, number) {
  const sound = job.kind === "music" || job.kind === "voice";
  const stage = el("div", { className: sound ? "stage sound" : "stage" });
  if (!sound && job.width) stage.style.aspectRatio = `${job.width} / ${job.height}`;
  const src = job.file ? mediaUrl(job) : null;
  const done = job.status === "done";

  if (done && job.media === "audio") {
    stage.append(el("audio", { src, controls: true, preload: "metadata" }));
  } else if (done && job.media === "video") {
    stage.append(el("video", { src, controls: true, loop: job.kind !== "movie", playsInline: true, preload: "metadata" }));
  } else if (done) {
    const img = el("img", { src, alt: job.prompt, loading: "lazy", className: "zoomable", title: "View full size" });
    img.onclick = () => { $("viewerImg").src = src; $("viewerImg").alt = job.prompt; $("viewer").showModal(); };
    stage.append(img);
  } else if (job.status === "error") {
    stage.append(el("div", { className: "status bad", textContent: job.error || "This take failed." }));
  } else {
    const waiting = job.kind === "movie" ? "Waiting for the cutting room…" : "Waiting for the camera…";
    const working = job.kind === "movie" ? "Cutting" : sound ? "Recording" : "Rolling";
    const label = job.status === "queued" ? waiting : `${working}… ${Math.round(job.progress * 100)}%`;
    const bar = el("div", { className: "bar" });
    bar.style.width = `${job.progress * 100}%`;
    stage.append(el("div", { className: "status", textContent: label }), bar);
  }

  const info = [`Take ${number}`, job.model_name];
  if (job.aspect) info.push(job.aspect);
  if (job.kind === "movie") info.push(`${job.clips.length} take${job.clips.length === 1 ? "" : "s"}`);
  if (job.seconds) info.push(`${job.seconds} s`);
  if (job.voice) info.push(job.voice, `pace ${job.speed}×`);
  if (job.lyrics) info.push("with lyrics");
  if (job.kind === "movie" && job.voice_file) info.push("voiceover");
  if (job.kind === "movie" && job.music_file) info.push("soundtrack");
  if (job.start_take) info.push("from start frame");
  if (job.seed !== undefined && job.kind !== "voice") info.push(`seed ${job.seed}`);
  if (job.elapsed) info.push(`${job.elapsed} s render`);

  const tools = el("div", { className: "tools" });
  const button = (text, onclick) => tools.append(el("button", { type: "button", textContent: text, onclick }));
  if (done && job.media === "image") button("Animate", () => animate(job));
  if (done && !sound) button("Add to movie", () => { state.cut.clips.push(job.id); renderCut(); });
  if (done && job.kind === "voice") button("Use as voiceover", () => { state.cut.voice = job.id; renderCut(); });
  if (done && job.kind === "music") button("Use as soundtrack", () => { state.cut.music = job.id; renderCut(); });
  if (src && done) tools.append(el("a", { href: src, download: job.file, textContent: "Download" }));
  if (!job.uploaded && job.kind !== "movie") button("Reuse", () => {
    $("prompt").value = job.prompt;
    if (job.seed !== undefined) $("seed").value = job.seed;
    state.kind = job.kind; state.model = job.model;
    state.start = job.start_take && state.jobs.has(job.start_take) ? job.start_take : null;
    renderControls();
    if (job.kind === "music") $("lyrics").value = job.lyrics || "";
    if (job.kind === "voice") { $("voice").value = job.voice; $("speed").value = String(job.speed); }
    $("prompt").focus();
  });
  if (job.status !== "running") button(job.status === "queued" ? "Cancel" : "Delete", async () => {
    try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); } catch (e) { return showError(e.message); }
    state.jobs.delete(job.id);
    renderControls();
    renderCut();
    renderTakes();
  });

  const meta = el("div", { className: "meta" }, el("p", { textContent: job.prompt }));
  if (done && job.placeholder) meta.append(el("span", { className: "tag", textContent: "Placeholder, not model output" }));
  if (job.note) meta.append(el("p", { className: "note", textContent: job.note }));
  meta.append(el("div", { className: "info", textContent: info.join(" · ") }), tools);
  return el("article", { className: "take" }, stage, meta);
}

const shown = (job) => state.show === "all" || job.kind === state.show
  || (state.show === "sound" && (job.kind === "music" || job.kind === "voice"));

let lastSignature = "";
function renderTakes() {
  const jobs = [...state.jobs.values()].sort((a, b) => b.created - a.created);
  // Skip the rebuild when nothing changed, so playing media is not reset by polling.
  const signature = state.show + "#" + jobs.map((j) => `${j.id}:${j.status}:${j.progress}`).join("|");
  if (signature === lastSignature) return;
  lastSignature = signature;
  const cards = [];
  jobs.forEach((job, i) => { if (shown(job)) cards.push(takeCard(job, jobs.length - i)); });
  $("takes").replaceChildren(...cards);
  $("empty").hidden = cards.length > 0;
  $("empty").textContent = jobs.length
    ? "Nothing of that kind in the reel yet."
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
  const text = $("prompt").value.trim();
  if (!text) return showError(`Write a ${WORDING[state.kind][0].toLowerCase()} first.`);
  const model = currentModel();
  const visual = VISUAL.has(state.kind);
  const body = { model: state.model, prompt: text };
  if (visual) {
    body.prompt = [text, $("look").value, ...state.directions].filter(Boolean).join(", ");
    body.aspect = $("aspect").value;
    body.negative_prompt = model.supports_negative ? $("negative").value : "";
    body.seconds = $("seconds").value;
    body.count = state.kind === "image" ? $("count").value : 1;
    body.start_take = state.kind === "video" && model.supports_start_image ? state.start : null;
  }
  if (state.kind === "music") { body.lyrics = $("lyrics").value; body.seconds = $("musicSeconds").value; }
  if (state.kind === "voice") { body.voice = $("voice").value; body.speed = $("speed").value; }
  if (state.kind !== "voice") body.seed = $("seed").value;

  $("action").disabled = true;
  try {
    const { jobs } = await postJson("/api/generate", body);
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
  $("engine").textContent = state.config.ready.visual ? "Local GPU engine" : "Preview engine";
  if (state.config.engine_note) {
    $("notice").textContent = state.config.engine_note;
    $("notice").hidden = false;
  }
  fillSelect($("count"), Array.from({ length: state.config.max_batch }, (_, i) => [i + 1, String(i + 1)]));
  fillSelect($("look"), LOOKS);
  fillSelect($("musicSeconds"), [15, 30, 60, 90, 120].map((s) => [s, s < 60 ? `${s} s` : `${s / 60} min`]));
  $("musicSeconds").value = "30";
  fillSelect($("speed"), [["0.8", "Slower"], ["1", "Normal"], ["1.2", "Faster"]]);
  $("speed").value = "1";
  renderChips();
  renderControls();
  renderCut();
  renderTakes();
  poll();
})();
