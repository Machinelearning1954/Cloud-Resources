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
  battles: [], leaderboard: {}, dismissed: new Set(),
  justVoted: new Set(),   // battles voted in this visit stay on screen so the reveal can be read
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
  const battle = visual && $("battle").checked;
  $("battleField").hidden = !visual;
  $("modelField").hidden = battle;
  $("action").textContent = battle ? "Start battle" : "Action!";

  $("aspectField").hidden = !visual;
  $("dirField").hidden = !visual;
  $("lookField").hidden = !visual;
  $("secondsField").hidden = kind !== "video";
  $("countField").hidden = kind !== "image" || battle;
  $("lyricsField").hidden = kind !== "music";
  $("contextField").hidden = kind === "voice";   // voiceovers speak the scene itself
  if (kind === "voice" && dictation) dictation.stop();
  $("musicSecondsField").hidden = kind !== "music";
  $("voiceField").hidden = kind !== "voice";
  $("speedField").hidden = kind !== "voice";
  $("more").hidden = kind === "voice" || battle;
  $("negativeField").hidden = !model.supports_negative;

  if (visual) fillSelect($("aspect"), state.config.aspects[kind].map((a) => [a, a]), true);
  if (kind === "video") {
    const longest = battle
      ? Math.min(...models.map((m) => m.max_seconds))   // both sides must be able to run the same length
      : model.max_seconds;
    const lengths = [];
    for (let s = 2; s <= longest; s += s < 6 ? 1 : 2) lengths.push([s, `${s} s`]);
    fillSelect($("seconds"), lengths, true);
    if (!$("seconds").dataset.touched) $("seconds").value = String(Math.min(5, longest));
  }
  if (kind === "voice") fillSelect($("voice"), model.voices, true);

  // Start frame: only for video models that can animate a picture.
  const canStart = kind === "video" && (battle || model.supports_start_image);
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

/* ---------- Arena ---------- */

async function loadArena() {
  try {
    const data = await api("/api/arena");
    state.battles = data.battles;
    state.leaderboard = data.leaderboard;
  } catch { /* keep what we have */ }
  renderArena(true);
}

function sideStage(job) {
  const stage = el("div", { className: "stage" });
  if (!job) {
    stage.append(el("div", { className: "status bad", textContent: "This take was deleted." }));
    return stage;
  }
  stage.style.aspectRatio = `${job.width} / ${job.height}`;
  if (job.status === "done" && job.media === "video") {
    stage.append(el("video", { src: mediaUrl(job), controls: true, loop: true, muted: true, playsInline: true, preload: "metadata" }));
  } else if (job.status === "done") {
    stage.append(el("img", { src: mediaUrl(job), alt: "", loading: "lazy" }));
  } else if (job.status === "error") {
    stage.append(el("div", { className: "status bad", textContent: job.error || "This take failed." }));
  } else {
    const bar = el("div", { className: "bar" });
    bar.style.width = `${job.progress * 100}%`;
    const label = job.status === "queued" ? "Waiting…" : `Rolling… ${Math.round(job.progress * 100)}%`;
    stage.append(el("div", { className: "status", textContent: label }), bar);
  }
  return stage;
}

let arenaSignature = "";
function renderArena(force) {
  const list = state.battles.filter((b) => !state.dismissed.has(b.id) && (!b.winner || state.justVoted.has(b.id)));
  const signature = list.map((b) => [b.id, b.winner, ...["a", "b"].map((s) => {
    const job = state.jobs.get(b[s]);
    return job ? `${job.status}:${job.progress}` : "gone";
  })].join(":")).join("|");
  if (!force && signature === arenaSignature) return;
  arenaSignature = signature;

  const hasBoard = Object.keys(state.leaderboard).length > 0;
  $("arena").hidden = list.length === 0 && !hasBoard;

  $("battles").replaceChildren(...list.map((b) => {
    const takes = { a: state.jobs.get(b.a), b: state.jobs.get(b.b) };
    const ready = takes.a?.status === "done" && takes.b?.status === "done";
    const sides = el("div", { className: "sides" }, ...["a", "b"].map((s) => {
      const name = b.winner ? `${s.toUpperCase()} · ${b.models[s]}` : `Take ${s.toUpperCase()}`;
      const side = el("div", { className: b.winner === s ? "side won" : "side" }, el("span", { textContent: name }), sideStage(takes[s]));
      return side;
    }));
    const votes = el("div", { className: "votes" });
    if (b.winner) {
      const verdict = b.winner === "tie" ? "You called it a tie." : `You picked ${b.winner.toUpperCase()}: ${b.models[b.winner]}.`;
      votes.append(verdict + (b.counts ? "" : " Practice round: placeholder takes do not count on the leaderboard."));
      votes.append(el("button", { type: "button", className: "link", textContent: "Dismiss", onclick: () => {
        state.dismissed.add(b.id);
        renderArena(true);
      } }));
    } else {
      for (const [value, text] of [["a", "A is better"], ["tie", "Tie"], ["b", "B is better"]]) {
        votes.append(el("button", { type: "button", className: "pick", textContent: text, disabled: !ready, onclick: async () => {
          try {
            await postJson(`/api/battles/${b.id}/vote`, { winner: value });
          } catch (e) { return showError(e.message); }
          state.justVoted.add(b.id);
          const { jobs } = await api("/api/jobs");          // model names are revealed now
          state.jobs = new Map(jobs.map((j) => [j.id, j]));
          lastSignature = "";
          renderTakes();
          await loadArena();
        } }));
      }
      if (!ready) votes.append("Voting opens when both takes finish.");
      votes.append(el("button", { type: "button", className: "link", textContent: "Discard", onclick: async () => {
        try { await api(`/api/battles/${b.id}`, { method: "DELETE" }); } catch (e) { return showError(e.message); }
        const { jobs } = await api("/api/jobs");
        state.jobs = new Map(jobs.map((j) => [j.id, j]));
        lastSignature = "";
        renderTakes();
        await loadArena();
      } }));
    }
    const duel = el("div", { className: "duel" }, el("p", { textContent: b.prompt }));
    if (b.context) duel.append(el("p", { className: "ctx", textContent: `Context: ${b.context}` }));
    duel.append(sides, votes);
    return duel;
  }));

  const tables = Object.entries(state.leaderboard).map(([kind, rows]) => el("table", {},
    el("caption", { textContent: `${kind} models` }),
    el("thead", {}, el("tr", {}, ...["#", "Model", "Rating", "Won", "Lost", "Tied"].map((h) => el("th", { textContent: h })))),
    el("tbody", {}, ...rows.map((r, i) => el("tr", {}, ...[i + 1, r.name, r.rating, r.wins, r.losses, r.ties]
      .map((v) => el("td", { textContent: String(v) })))))));
  $("boardBody").replaceChildren(...(tables.length ? tables
    : [el("p", { textContent: "No counted votes yet. Votes count once the takes are real model output." })]));
}

/* ---------- Context box and speech input ---------- */

// The browser's own speech recogniser: free, and no model to install. Chrome and Edge send the
// audio to their speech service; Firefox has no recogniser, so the button is disabled there.
const SpeechRecognizer = window.SpeechRecognition || window.webkitSpeechRecognition || null;
const SPEECH_ERRORS = {
  "not-allowed": "The microphone is blocked. Allow it for this page, on localhost or an https address.",
  "service-not-allowed": "This browser does not allow speech recognition here.",
  "audio-capture": "No microphone was found.",
  network: "The browser's speech service could not be reached. Check the connection and try again.",
  "no-speech": "No speech was heard. Press Speak and try again.",
};
let dictation = null;   // the running recogniser, or null while the microphone is off

// tone: "live" while listening, "warn" for a limit, "bad" for an error, or "" for plain information.
function setContextStatus(text, tone = "") {
  const status = $("contextStatus");
  status.textContent = text || "";
  status.className = `hint status ${tone}`.trim();
  status.hidden = !text;
}

function setMicOn(on) {
  $("contextMic").setAttribute("aria-pressed", String(on));
  $("contextMicLabel").textContent = on ? "Stop" : "Speak";
  $("context").readOnly = on;   // the words are being written in; typing now would be overwritten
  $("context").classList.toggle("listening", on);
}

// The counter shows how much of the 1000-character limit is used, so it is not a surprise on submit.
function updateContextCount() {
  const box = $("context");
  const count = $("contextCount");
  const used = box.value.length;
  count.textContent = `${used} / ${box.maxLength}`;
  count.classList.toggle("near", used >= box.maxLength * 0.9 && used < box.maxLength);
  count.classList.toggle("full", used >= box.maxLength);
}

// Drops the microphone at once. Anything the recogniser still sends is ignored, so it cannot
// overwrite text that was put in the box meanwhile (for example by Reuse).
function cancelDictation() {
  const rec = dictation;
  if (!rec) return;
  dictation = null;
  rec.abort();
  setMicOn(false);
  setContextStatus("");
}

function toggleDictation() {
  if (dictation) return dictation.stop();   // the last phrase still arrives, then onend runs
  const box = $("context");
  const before = box.value.slice(0, box.selectionStart);
  const after = box.value.slice(box.selectionEnd);
  let committed = "";      // finished phrases; the interim guess is shown but not kept
  let failed = false;
  let full = false;
  const rec = new SpeechRecognizer();
  rec.lang = navigator.language || "en-US";
  rec.continuous = true;
  rec.interimResults = true;
  const build = (words) => {
    const lead = words && before && !/\s$/.test(before) ? " " : "";
    const tail = words && after && !/^\s/.test(after) ? " " : "";
    return { text: before + lead + words + tail + after, caret: (before + lead + words).length };
  };
  rec.onresult = (event) => {
    if (dictation !== rec) return;   // cancelled: ignore what is still arriving
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) committed += result[0].transcript;
      else interim += result[0].transcript;
    }
    let spoken = (committed + interim).trim();
    let written = build(spoken);
    // Typed text is capped by maxlength, but a value set in code is not, so cap the speech here.
    const excess = written.text.length - box.maxLength;
    if (excess > 0) {
      spoken = spoken.slice(0, Math.max(0, spoken.length - excess)).trimEnd();
      written = build(spoken);
      full = true;
      setContextStatus("The context box is full, so dictation has stopped.", "warn");
      rec.stop();
    }
    box.value = written.text;
    box.setSelectionRange(written.caret, written.caret);
    updateContextCount();
  };
  rec.onerror = (event) => {
    if (dictation !== rec) return;
    failed = true;
    setContextStatus(SPEECH_ERRORS[event.error] || `Speech stopped (${event.error}).`, "bad");
  };
  rec.onend = () => {
    if (dictation !== rec) return;   // cancelled, or a newer dictation has started
    dictation = null;
    setMicOn(false);
    if (!failed && !full) setContextStatus("");
  };
  dictation = rec;
  setMicOn(true);
  setContextStatus("Listening. Press Stop when you are done.", "live");
  try {
    rec.start();
  } catch (e) {
    dictation = null;
    setMicOn(false);
    setContextStatus(`Could not start listening: ${e.message}`, "bad");
  }
}

function setupSpeech() {
  if (SpeechRecognizer) return;
  $("contextMic").disabled = true;
  $("contextMic").title = "Speech input needs Chrome, Edge or Safari. Type your context instead.";
  $("contextHint").textContent = "Speech input needs Chrome, Edge or Safari. Type your context here instead.";
}

$("contextMic").onclick = toggleDictation;
$("context").addEventListener("input", updateContextCount);

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
  if (!job.uploaded && job.kind !== "movie" && !job.masked) button("Reuse", () => {
    cancelDictation();
    $("prompt").value = job.prompt;
    $("context").value = job.context || "";
    updateContextCount();
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
    if (job.battle) loadArena();
  });

  const meta = el("div", { className: "meta" }, el("p", { textContent: job.prompt }));
  if (job.context) meta.append(el("p", { className: "ctx", textContent: `Context: ${job.context}` }));
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
      renderArena();
    } catch { /* server briefly unreachable; try again on the next tick */ }
  }
  setTimeout(poll, 1000);
}

$("slate").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  if (dictation && state.kind !== "voice") return showError("Press Stop first, so the last words reach the context box.");
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
  if (state.kind !== "voice") body.context = $("context").value.trim();
  if (state.kind !== "voice") body.seed = $("seed").value;

  const battle = visual && $("battle").checked;
  if (battle) body.kind = state.kind;
  $("action").disabled = true;
  try {
    const { jobs } = await postJson(battle ? "/api/battle" : "/api/generate", body);
    jobs.forEach((j) => state.jobs.set(j.id, j));
    renderTakes();
    if (battle) await loadArena();
  } catch (e) {
    showError(e.message);
  } finally {
    $("action").disabled = false;
  }
});

for (const id of ["prompt", "context"]) {
  $(id).addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("slate").requestSubmit();
  });
}

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
$("battle").addEventListener("change", renderControls);
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
  setupSpeech();
  updateContextCount();
  renderControls();
  renderCut();
  renderTakes();
  await loadArena();
  poll();
})();
