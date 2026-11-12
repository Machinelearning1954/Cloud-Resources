"""Yardi Movies Director - a free, self-hosted AI image and video studio.

No accounts, no subscriptions, no credit metering: generations run on your own
hardware with open-weights models, so the only limit is your GPU.
"""
from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory

import arena
import audio
import movie
from engines import pick_engine
from registry import ASPECTS, MODELS

APP_NAME = "Yardi Movies Director"
MAX_PROMPT = 2000
MAX_SCRIPT = 5000
MAX_BATCH = 4
MAX_CLIPS = 60
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_SIDE = 2048


def create_app(out_dir: str | None = None, engine=None) -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.environ.get("YARDI_OUTPUT_DIR", os.path.join(here, "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    library_path = os.path.join(out_dir, "library.json")
    max_queue = int(os.environ.get("YARDI_MAX_QUEUE", "200"))

    app = Flask(__name__, static_folder=os.path.join(here, "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    engine = engine or pick_engine()
    lock = threading.Lock()
    jobs: dict[str, dict] = {}
    pending: queue.Queue[str] = queue.Queue()

    if os.path.exists(library_path):
        try:
            with open(library_path) as f:
                for job in json.load(f):
                    jobs[job["id"]] = job
        except (OSError, ValueError):
            app.logger.warning("Could not read %s; starting with an empty library", library_path)

    battles_path = os.path.join(out_dir, "battles.json")
    battles: dict[str, dict] = {}
    if os.path.exists(battles_path):
        try:
            with open(battles_path) as f:
                for battle in json.load(f):
                    # An unvoted battle is only useful while both of its takes still exist.
                    if battle.get("winner") or (battle["a"] in jobs and battle["b"] in jobs):
                        battles[battle["id"]] = battle
        except (OSError, ValueError, KeyError):
            app.logger.warning("Could not read %s; starting with no battles", battles_path)

    def save_battles():
        tmp = battles_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(list(battles.values()), f)
        os.replace(tmp, battles_path)

    def shown(job: dict) -> dict:
        """A take as the browser may see it: the model stays hidden while its battle is unvoted."""
        view = dict(job)
        battle = battles.get(job.get("battle"))
        if battle and not battle.get("winner"):
            view.update(model=None, model_name="Hidden until you vote", masked=True)
        return view

    def battle_view(battle: dict) -> dict:
        view = {k: battle.get(k) for k in ("id", "kind", "prompt", "a", "b", "winner", "counts", "created")}
        if battle.get("winner"):
            view["models"] = {s: MODELS[battle[s + "_model"]].name for s in ("a", "b")}
        return view

    def save_library():
        done = [j for j in jobs.values() if j["status"] == "done"]
        tmp = library_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(done, f)
        os.replace(tmp, library_path)

    def worker():
        while True:
            job_id = pending.get()
            with lock:
                job = jobs.get(job_id)
                if job is None:        # deleted while queued
                    continue
                job["status"] = "running"
            def progress(p: float, job=job):
                job["progress"] = round(max(0.0, min(1.0, p)), 3)

            started = time.time()
            try:
                if job["kind"] == "movie":
                    result = movie.cut(movie.ffmpeg_path(), out_dir, job["id"], job, progress)
                else:
                    result = engine.generate(MODELS[job["model"]], job, out_dir, job["id"], progress)
                with lock:
                    job.update(result, status="done", progress=1.0, elapsed=round(time.time() - started, 1))
                    save_library()
            except Exception as exc:  # surface the failure to the UI instead of killing the worker
                app.logger.exception("Generation failed for job %s", job_id)
                with lock:
                    job.update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, daemon=True, name="yardi-worker").start()

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/config")
    def config():
        gpu = engine.name != "preview"
        notes = []
        if not gpu:
            notes.append(
                "No GPU engine is loaded, so pictures, video and music are placeholders, not model output. "
                "Install requirements-gpu.txt on a CUDA machine for real generations."
            )
        if not audio.voice_ready():
            notes.append(
                "Voiceovers are placeholder tones until the voice model is installed "
                "(requirements-voice.txt). It runs on an ordinary CPU, no GPU needed."
            )
        if not movie.ffmpeg_path():
            notes.append("Install ffmpeg to cut takes into a movie.")
        return jsonify(
            name=APP_NAME,
            engine=engine.name,
            engine_note=" ".join(notes),
            ready={"visual": gpu, "music": gpu, "voice": audio.voice_ready(), "movie": bool(movie.ffmpeg_path())},
            models=[m.public() for m in MODELS.values()],
            aspects={kind: list(sizes) for kind, sizes in ASPECTS.items()},
            plan={"price": 0, "credits": "unlimited", "subscription": None},
            max_batch=MAX_BATCH,
        )

    @app.post("/api/generate")
    def generate():
        body = request.get_json(silent=True) or {}
        spec = MODELS.get(body.get("model"))
        if spec is None:
            return jsonify(error="Unknown model."), 400
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return jsonify(error="Write a prompt first."), 400
        limit = MAX_SCRIPT if spec.kind == "voice" else MAX_PROMPT
        if len(prompt) > limit:
            return jsonify(error=f"That is longer than {limit} characters."), 400
        aspect, width, height = "", None, None
        if spec.kind in ASPECTS:
            aspect = body.get("aspect") or "16:9"
            if aspect not in ASPECTS[spec.kind]:
                return jsonify(error=f"Aspect {aspect} is not available for {spec.kind}."), 400
            width, height = spec.sizes.get(aspect) or ASPECTS[spec.kind][aspect]
        voice = body.get("voice") or (spec.voices[0][0] if spec.voices else None)
        if spec.kind == "voice" and voice not in dict(spec.voices):
            return jsonify(error="Unknown voice."), 400
        try:
            count = int(body.get("count") or 1) if spec.kind == "image" else 1
            seconds = float(body.get("seconds") or (30 if spec.kind == "music" else 5))
            speed = float(body.get("speed") or 1)
            seed = body.get("seed")
            seed = int(seed) if seed not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify(error="count, seconds, speed and seed must be numbers."), 400
        count = max(1, min(MAX_BATCH, count))
        start_id = body.get("start_take")
        if start_id and not spec.i2v_pipeline:
            return jsonify(error=f"{spec.name} cannot start from an image."), 400
        if seed is not None and not 0 <= seed < 2**32:
            return jsonify(error="Seed must be between 0 and 4294967295."), 400

        created = []
        with lock:
            start = jobs.get(start_id) if start_id else None
            if start_id and not (start and start["status"] == "done" and start.get("media") == "image"):
                return jsonify(error="That start frame is no longer in the reel."), 400
            waiting = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
            if waiting + count > max_queue:
                return jsonify(error="The render queue is full. Let a few takes finish first."), 429
            for i in range(count):
                job = {
                    "id": uuid.uuid4().hex,
                    "model": spec.id,
                    "model_name": spec.name,
                    "kind": spec.kind,
                    "prompt": prompt,
                    "negative_prompt": str(body.get("negative_prompt") or "")[:MAX_PROMPT],
                    "aspect": aspect,
                    "width": width,
                    "height": height,
                    "seed": (seed + i) % 2**32 if seed is not None else random.randrange(2**32),
                    "status": "queued",
                    "progress": 0.0,
                    "created": time.time(),
                }
                if spec.kind == "video":
                    job["seconds"] = max(1.0, min(float(spec.max_seconds), seconds))
                if spec.kind == "music":
                    job["seconds"] = max(5.0, min(float(spec.max_seconds), seconds))
                    job["lyrics"] = str(body.get("lyrics") or "").strip()[:MAX_SCRIPT]
                if spec.kind == "voice":
                    job["voice"], job["speed"] = voice, max(0.5, min(2.0, speed))
                if start:
                    job["start_take"], job["start_file"] = start["id"], start["file"]
                jobs[job["id"]] = job
                created.append(dict(job))
        for job in created:
            pending.put(job["id"])
        return jsonify(jobs=created), 202

    @app.post("/api/battle")
    def start_battle():
        """Same prompt and seed on two different models, picked at random, names hidden."""
        body = request.get_json(silent=True) or {}
        kind = body.get("kind")
        if kind not in ASPECTS:
            return jsonify(error="Battles are for image or video models."), 400
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return jsonify(error="Write a prompt first."), 400
        if len(prompt) > MAX_PROMPT:
            return jsonify(error=f"That is longer than {MAX_PROMPT} characters."), 400
        aspect = body.get("aspect") or "16:9"
        if aspect not in ASPECTS[kind]:
            return jsonify(error=f"Aspect {aspect} is not available for {kind}."), 400
        try:
            seconds = float(body.get("seconds") or 5)
            seed = body.get("seed")
            seed = int(seed) if seed not in (None, "") else random.randrange(2**32)
        except (TypeError, ValueError):
            return jsonify(error="seconds and seed must be numbers."), 400
        if not 0 <= seed < 2**32:
            return jsonify(error="Seed must be between 0 and 4294967295."), 400
        start_id = body.get("start_take")
        pool = [m for m in MODELS.values() if m.kind == kind and (not start_id or m.i2v_pipeline)]
        if len(pool) < 2:
            return jsonify(error="A battle needs two models that can do this."), 400
        pair = random.sample(pool, 2)

        with lock:
            start = jobs.get(start_id) if start_id else None
            if start_id and not (start and start["status"] == "done" and start.get("media") == "image"):
                return jsonify(error="That start frame is no longer in the reel."), 400
            if sum(1 for j in jobs.values() if j["status"] in ("queued", "running")) + 2 > max_queue:
                return jsonify(error="The render queue is full. Let a few takes finish first."), 429
            battle_id = uuid.uuid4().hex
            now = time.time()
            takes = []
            for offset, spec in enumerate(pair):
                width, height = spec.sizes.get(aspect) or ASPECTS[kind][aspect]
                job = {
                    "id": uuid.uuid4().hex, "model": spec.id, "model_name": spec.name, "kind": kind,
                    "prompt": prompt, "negative_prompt": "", "aspect": aspect, "width": width, "height": height,
                    "seed": seed, "battle": battle_id,
                    "status": "queued", "progress": 0.0, "created": now + offset / 1000,
                }
                if kind == "video":
                    # Both sides get the same length: the shorter of the two models' limits.
                    job["seconds"] = max(1.0, min(seconds, *(float(m.max_seconds) for m in pair)))
                if start:
                    job["start_take"], job["start_file"] = start["id"], start["file"]
                jobs[job["id"]] = job
                takes.append(job)
            battle = {
                "id": battle_id, "kind": kind, "prompt": prompt, "created": now, "winner": None,
                "a": takes[0]["id"], "b": takes[1]["id"], "a_model": pair[0].id, "b_model": pair[1].id,
            }
            battles[battle_id] = battle
            save_battles()
            reply = jsonify(battle=battle_view(battle), jobs=[shown(j) for j in takes])
        for job in takes:
            pending.put(job["id"])
        return reply, 202

    @app.get("/api/arena")
    def arena_state():
        with lock:
            ordered = sorted(battles.values(), key=lambda b: b["created"], reverse=True)
            board = arena.leaderboard(ordered, {m.id: m.name for m in MODELS.values()})
            return jsonify(battles=[battle_view(b) for b in ordered], leaderboard=board)

    @app.post("/api/battles/<battle_id>/vote")
    def vote(battle_id):
        winner = (request.get_json(silent=True) or {}).get("winner")
        if winner not in arena.SCORES:
            return jsonify(error="Vote a, b or tie."), 400
        with lock:
            battle = battles.get(battle_id)
            if battle is None:
                return jsonify(error="No such battle."), 404
            if battle["winner"]:
                return jsonify(error="This battle already has a vote."), 409
            sides = [jobs.get(battle["a"]), jobs.get(battle["b"])]
            if not all(j and j["status"] == "done" for j in sides):
                return jsonify(error="Both takes have to finish before you can vote."), 409
            # Placeholder takes are not model output, so a vote on them says nothing about the models.
            battle.update(winner=winner, voted_at=time.time(), counts=not any(j.get("placeholder") for j in sides))
            save_battles()
            return jsonify(battle_view(battle))

    @app.delete("/api/battles/<battle_id>")
    def discard_battle(battle_id):
        """Drop an unvoted battle. Its takes stay in the reel with their models revealed."""
        with lock:
            battle = battles.get(battle_id)
            if battle is None:
                return jsonify(error="No such battle."), 404
            if battle["winner"]:
                return jsonify(error="A voted battle is part of the leaderboard and cannot be discarded."), 409
            del battles[battle_id]
            save_battles()
        return jsonify(ok=True)

    @app.post("/api/movie")
    def make_movie():
        """Join takes, in order, into one MP4 with an optional voiceover and soundtrack."""
        if not movie.ffmpeg_path():
            return jsonify(error="ffmpeg is not installed, so movies cannot be cut on this machine."), 400
        body = request.get_json(silent=True) or {}
        clip_ids = body.get("clips")
        if not isinstance(clip_ids, list) or not clip_ids:
            return jsonify(error="Add at least one picture or video take to the movie."), 400
        if len(clip_ids) > MAX_CLIPS:
            return jsonify(error=f"A movie can hold up to {MAX_CLIPS} takes."), 400
        try:
            volume = max(0.0, min(1.0, float(body.get("music_volume", 0.35))))
            still = max(1.0, min(15.0, float(body.get("still_seconds", 3))))
        except (TypeError, ValueError):
            return jsonify(error="music_volume and still_seconds must be numbers."), 400

        with lock:
            def done(job_id, medias):
                take = jobs.get(job_id) if isinstance(job_id, str) else None
                return take if take and take["status"] == "done" and take.get("media") in medias else None

            clips = [done(i, ("image", "video", "animation")) for i in clip_ids]
            if not all(clips):
                return jsonify(error="A take in this movie is no longer in the reel."), 400
            tracks = {}
            for key in ("voice", "music"):
                if body.get(key):
                    tracks[key] = done(body[key], ("audio",))
                    if not tracks[key]:
                        return jsonify(error=f"That {key} take is no longer in the reel."), 400
            if sum(1 for j in jobs.values() if j["status"] in ("queued", "running")) >= max_queue:
                return jsonify(error="The render queue is full. Let a few takes finish first."), 429
            used = clips + list(tracks.values())
            title = str(body.get("title") or "").strip()[:120]
            job = {
                "id": uuid.uuid4().hex, "model": "movie", "model_name": "Movie cut", "kind": "movie",
                "prompt": title or f"Movie: {len(clips)} take{'s' if len(clips) != 1 else ''}",
                "aspect": "", "width": clips[0]["width"], "height": clips[0]["height"],
                "clips": [{"id": c["id"], "file": c["file"], "media": c["media"]} for c in clips],
                "voice_file": tracks["voice"]["file"] if "voice" in tracks else None,
                "music_file": tracks["music"]["file"] if "music" in tracks else None,
                "music_volume": volume, "still_seconds": still,
                "placeholder": any(t.get("placeholder") for t in used),
                "status": "queued", "progress": 0.0, "created": time.time(),
            }
            jobs[job["id"]] = job
        pending.put(job["id"])
        return jsonify(dict(job)), 202

    @app.post("/api/upload")
    def upload():
        """Add your own picture to the reel so it can be used as a start frame."""
        from PIL import Image, UnidentifiedImageError

        sent = request.files.get("image")
        if sent is None:
            return jsonify(error="Choose an image file."), 400
        try:
            with Image.open(sent.stream) as img:
                img = img.convert("RGB")
                img.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE))
                job_id = uuid.uuid4().hex
                img.save(os.path.join(out_dir, job_id + ".png"))   # re-encoded: never store the raw upload
                width, height = img.size
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="That file is not an image this studio can read."), 400
        job = {
            "id": job_id, "model": "upload", "model_name": "Your upload", "kind": "image",
            "prompt": os.path.basename(sent.filename or "")[:120] or "Uploaded frame",
            "aspect": f"{width}×{height}", "width": width, "height": height,
            "status": "done", "progress": 1.0, "created": time.time(),
            "file": job_id + ".png", "media": "image", "uploaded": True,
        }
        with lock:
            jobs[job_id] = job
            save_library()
        return jsonify(dict(job)), 201

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error=f"Uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            items = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)
            return jsonify(jobs=[shown(j) for j in items])

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        with lock:
            job = jobs.get(job_id)
            return (jsonify(shown(job)), 200) if job else (jsonify(error="No such take."), 404)

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify(error="No such take."), 404
            if job["status"] == "running":
                return jsonify(error="This take is rendering. Delete it once it finishes."), 409
            del jobs[job_id]
            save_library()
            battle = battles.get(job.get("battle"))
            if battle and not battle["winner"]:      # a battle cannot be judged with one side missing
                del battles[battle["id"]]
                save_battles()
        if job.get("file"):
            try:
                os.remove(os.path.join(out_dir, job["file"]))
            except OSError:
                pass
        return jsonify(ok=True)

    @app.get("/media/<path:name>")
    def media(name):
        return send_from_directory(out_dir, name)

    @app.get("/health")
    def health():
        return jsonify(status="ok", engine=engine.name)

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("YARDI_HOST", "127.0.0.1"),
        port=int(os.environ.get("YARDI_PORT", "7860")),
        threaded=True,
    )
