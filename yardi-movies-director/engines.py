"""Generation engines.

DiffusersEngine  runs the real open-weights models on a local GPU.
PreviewEngine    draws labelled placeholder storyboards with Pillow so the whole
                 app can be used and tested on a machine with no GPU.
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import time
from typing import Callable

from registry import ModelSpec

Progress = Callable[[float], None]


def start_frame(req: dict, out_dir: str, size: tuple[int, int]):
    """Load the take chosen as the first frame, cropped to fill `size`. None for text-only jobs."""
    if not req.get("start_file"):
        return None
    from PIL import Image, ImageOps

    path = os.path.join(out_dir, req["start_file"])
    if not os.path.exists(path):
        raise RuntimeError("The start frame for this take was deleted before it rendered.")
    with Image.open(path) as img:
        return ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS)


class PreviewEngine:
    name = "preview"
    note = (
        "Preview mode: no GPU engine is loaded, so takes are placeholder storyboards, "
        "not model output. Install requirements-gpu.txt on a CUDA machine for real generations."
    )

    def generate(self, spec: ModelSpec, req: dict, out_dir: str, stem: str, progress: Progress) -> dict:
        from PIL import Image

        w, h = req["width"], req["height"]
        scale = 640 / max(w, h)
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        rng = random.Random(req["seed"])
        hue = int(hashlib.sha256(req["prompt"].encode()).hexdigest()[:4], 16) % 360
        scene = _Scene(w, h, hue, rng, start_frame(req, out_dir, (w, h)))

        if spec.kind == "image":
            for i in range(4):
                time.sleep(0.05)
                progress((i + 1) / 5)
            img = scene.frame(0.35, spec, req)
            name = stem + ".png"
            img.save(os.path.join(out_dir, name))
            return {"file": name, "media": "image"}

        n = 24
        frames = []
        for i in range(n):
            frames.append(scene.frame(i / (n - 1), spec, req))
            progress((i + 1) / (n + 1))
        name = stem + ".webp"
        frames[0].save(
            os.path.join(out_dir, name), save_all=True, append_images=frames[1:],
            duration=83, loop=0, quality=70,
        )
        return {"file": name, "media": "animation"}


class _Scene:
    """A tiny procedural landscape: sky gradient, sun, hills, slate caption."""

    def __init__(self, w: int, h: int, hue: int, rng: random.Random, still=None):
        from PIL import Image

        self.w, self.h = w, h
        self.still = still
        top, bottom = _hsl(hue, 45, 16), _hsl((hue + 40) % 360, 60, 46)
        sky = Image.new("RGB", (1, h))
        sky.putdata([tuple(int(top[c] + (bottom[c] - top[c]) * (y / h) ** 1.4) for c in range(3)) for y in range(h)])
        self.sky = sky.resize((w, h))
        self.sun = _hsl((hue + 50) % 360, 90, 78)
        self.hills = []
        for layer in range(3):
            base = h * (0.58 + 0.13 * layer)
            amp = h * (0.10 - 0.02 * layer)
            waves = [(rng.uniform(0.6, 2.4), rng.uniform(0, 6.28), rng.uniform(0.4, 1.0)) for _ in range(3)]
            self.hills.append((base, amp, waves, _hsl(hue, 35, 22 - 6 * layer), 18 * (layer + 1)))
        self.sun_x = rng.uniform(0.25, 0.75)

    def frame(self, t: float, spec: ModelSpec, req: dict):
        from PIL import ImageDraw, ImageFont

        w, h = self.w, self.h
        font = ImageFont.load_default(size=13)
        if self.still is not None:
            # Start frame given: a slow push-in on the still stands in for real motion.
            crop_w, crop_h = w * (1 - 0.16 * t), h * (1 - 0.16 * t)
            left, top = (w - crop_w) / 2, (h - crop_h) / 2
            img = self.still.resize((w, h), box=(left, top, left + crop_w, top + crop_h))
        else:
            img = self.sky.copy()
        d = ImageDraw.Draw(img)
        r = h * 0.11
        sx, sy = w * self.sun_x, h * (0.50 - 0.22 * t)
        if self.still is None:
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=self.sun)
        for base, amp, waves, colour, drift in self.hills if self.still is None else ():
            pts = [(0, h)]
            for x in range(0, w + 8, 8):
                u = (x + t * drift) / w
                y = base + amp * sum(a * math.sin(f * 6.28 * u + p) for f, p, a in waves) / 2
                pts.append((x, y))
            pts.append((w, h))
            d.polygon(pts, fill=colour)
        bar = max(26, h // 11)
        d.rectangle([0, 0, w, bar], fill=(12, 11, 10))
        d.rectangle([0, h - bar, w, h], fill=(12, 11, 10))
        d.text((10, bar // 2), f"PREVIEW · {spec.name} · not model output", fill=(232, 178, 84), font=font, anchor="lm")
        limit = max(12, (w - 20) // 7)
        caption = req["prompt"] if len(req["prompt"]) <= limit else req["prompt"][: limit - 1] + "…"
        d.text((10, h - bar // 2), caption, fill=(236, 230, 220), font=font, anchor="lm")
        return img


def _hsl(h: float, s: float, l: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return int(r * 255), int(g * 255), int(b * 255)


class DiffusersEngine:
    """Runs the registry's pipelines with Hugging Face diffusers on a local GPU.

    One pipeline is kept in memory at a time; switching models unloads the
    previous one so a single consumer GPU can serve all five.
    """

    name = "diffusers"
    note = "Running open-weights models locally. No credits, no metering."

    def __init__(self):
        import torch
        import diffusers

        self.torch, self.diffusers = torch, diffusers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offload = os.environ.get("YARDI_CPU_OFFLOAD", "1") != "0"
        self._pipe = None
        self._loaded = None

    def _load(self, spec: ModelSpec, from_image: bool = False):
        key = (spec.id, from_image)
        if self._loaded == key:
            return self._pipe
        self._pipe = None
        self._loaded = None
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        cls_name = spec.i2v_pipeline if from_image else spec.pipeline
        cls = getattr(self.diffusers, cls_name, None)
        if cls is None:
            raise RuntimeError(
                f"Your diffusers version has no {cls_name}. Upgrade with: pip install -U diffusers"
            )
        dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
        pipe = cls.from_pretrained(spec.repo, torch_dtype=dtype, **(spec.i2v_load if from_image else {}))
        if self.device == "cuda" and self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self._pipe, self._loaded = pipe, key
        return pipe

    def generate(self, spec: ModelSpec, req: dict, out_dir: str, stem: str, progress: Progress) -> dict:
        first = start_frame(req, out_dir, (req["width"], req["height"]))
        pipe = self._load(spec, from_image=first is not None)
        steps = spec.steps

        def on_step(_pipe, step, _timestep, kwargs):
            progress(min(0.97, (step + 1) / steps))
            return kwargs

        kwargs = dict(
            prompt=req["prompt"],
            width=req["width"],
            height=req["height"],
            num_inference_steps=steps,
            generator=self.torch.Generator("cpu").manual_seed(req["seed"]),
            callback_on_step_end=on_step,
            **spec.guidance,
        )
        if spec.supports_negative:
            # Qwen-Image only applies true CFG when a negative prompt is present.
            kwargs["negative_prompt"] = req.get("negative_prompt") or " "

        if spec.kind == "image":
            image = pipe(**kwargs).images[0]
            name = stem + ".png"
            image.save(os.path.join(out_dir, name))
            return {"file": name, "media": "image"}

        name = stem + ".mp4"
        path = os.path.join(out_dir, name)
        kwargs["num_frames"] = spec.frames_for(req["seconds"])
        if first is not None:
            kwargs["image"] = first
        if spec.has_audio:
            from diffusers.utils import encode_video

            video, audio = pipe(**kwargs, frame_rate=float(spec.fps), output_type="np", return_dict=False)
            encode_video(
                video[0], fps=spec.fps, output_path=path,
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            )
        else:
            from diffusers.utils import export_to_video

            frames = pipe(**kwargs).frames[0]
            export_to_video(frames, path, fps=spec.fps)
        return {"file": name, "media": "video"}


def pick_engine():
    """YARDI_ENGINE=auto|diffusers|preview. `auto` uses diffusers when a GPU stack is present."""
    choice = os.environ.get("YARDI_ENGINE", "auto").lower()
    if choice == "preview":
        return PreviewEngine()
    try:
        engine = DiffusersEngine()
    except ImportError:
        if choice == "diffusers":
            raise
        return PreviewEngine()
    if choice == "auto" and engine.device != "cuda":
        return PreviewEngine()
    return engine
