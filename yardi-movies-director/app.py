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

from engines import pick_engine
from registry import ASPECTS, MODELS

APP_NAME = "Yardi Movies Director"
MAX_PROMPT = 2000
MAX_BATCH = 4


def create_app(out_dir: str | None = None, engine=None) -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.environ.get("YARDI_OUTPUT_DIR", os.path.join(here, "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    library_path = os.path.join(out_dir, "library.json")
    max_queue = int(os.environ.get("YARDI_MAX_QUEUE", "200"))

    app = Flask(__name__, static_folder=os.path.join(here, "static"), static_url_path="/static")
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
            spec = MODELS[job["model"]]

            def progress(p: float, job=job):
                job["progress"] = round(max(0.0, min(1.0, p)), 3)

            started = time.time()
            try:
                result = engine.generate(spec, job, out_dir, job["id"], progress)
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
        return jsonify(
            name=APP_NAME,
            engine=engine.name,
            engine_note=engine.note,
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
        if len(prompt) > MAX_PROMPT:
            return jsonify(error=f"Prompt is longer than {MAX_PROMPT} characters."), 400
        aspect = body.get("aspect") or "16:9"
        if aspect not in ASPECTS[spec.kind]:
            return jsonify(error=f"Aspect {aspect} is not available for {spec.kind}."), 400
        width, height = ASPECTS[spec.kind][aspect]
        try:
            count = int(body.get("count") or 1)
            seconds = float(body.get("seconds") or 5)
            seed = body.get("seed")
            seed = int(seed) if seed not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify(error="count, seconds and seed must be numbers."), 400
        count = max(1, min(MAX_BATCH, count))
        if seed is not None and not 0 <= seed < 2**32:
            return jsonify(error="Seed must be between 0 and 4294967295."), 400

        created = []
        with lock:
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
                jobs[job["id"]] = job
                created.append(dict(job))
        for job in created:
            pending.put(job["id"])
        return jsonify(jobs=created), 202

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            items = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)
            return jsonify(jobs=[dict(j) for j in items])

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        with lock:
            job = jobs.get(job_id)
            return (jsonify(dict(job)), 200) if job else (jsonify(error="No such take."), 404)

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
