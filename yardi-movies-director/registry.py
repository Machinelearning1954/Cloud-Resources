"""Model registry for Yardi Movies Director.

Every model here is an open-weights model you run on your own hardware.
Repo ids can be overridden with environment variables (YARDI_REPO_<ID>), so a
local path or a mirror works without touching the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    kind: str                     # "image" | "video"
    licence: str
    repo: str
    pipeline: str                 # class name inside `diffusers`
    blurb: str
    steps: int
    guidance: dict = field(default_factory=dict)   # kwargs passed straight to the pipeline
    supports_negative: bool = False
    fps: int = 0                  # video only
    frame_step: int = 0           # video only: num_frames must be k * frame_step + 1
    max_seconds: int = 0          # video only
    has_audio: bool = False
    i2v_pipeline: str = ""        # video only: diffusers class used when a start frame is given
    i2v_load: dict = field(default_factory=dict)   # extra from_pretrained kwargs for that class
    default_negative: str = " "   # used when the user leaves "keep out of frame" empty
    sizes: dict = field(default_factory=dict)      # per-model (width, height) overrides by aspect

    def frames_for(self, seconds: float) -> int:
        raw = int(round(seconds * self.fps))
        k = max(1, raw // self.frame_step)
        return k * self.frame_step + 1

    def public(self) -> dict:
        d = asdict(self)
        for key in ("guidance", "pipeline", "i2v_pipeline", "i2v_load", "default_negative", "sizes"):
            d.pop(key)
        d["supports_start_image"] = bool(self.i2v_pipeline)
        return d


def _repo(model_id: str, default: str) -> str:
    return os.environ.get("YARDI_REPO_" + model_id.upper().replace("-", "_"), default)


_LTX_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"

MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        ModelSpec(
            id="z-image-turbo",
            name="Z-Image Turbo",
            kind="image",
            licence="Apache 2.0",
            repo=_repo("z-image-turbo", "Tongyi-MAI/Z-Image-Turbo"),
            pipeline="ZImagePipeline",
            blurb="Fast few-step stills. The quickest way to rough out a frame.",
            steps=9,
            guidance={"guidance_scale": 0.0},
        ),
        ModelSpec(
            id="qwen-image-2512",
            name="Qwen-Image-2512",
            kind="image",
            licence="Apache 2.0",
            repo=_repo("qwen-image-2512", "Qwen/Qwen-Image-2512"),
            pipeline="QwenImagePipeline",
            blurb="Detailed stills with strong text rendering. Good for posters and title cards.",
            steps=50,
            guidance={"true_cfg_scale": 4.0},
            supports_negative=True,
        ),
        ModelSpec(
            id="flux2-klein-4b",
            name="FLUX.2 klein 4B",
            kind="image",
            licence="Apache 2.0",
            repo=_repo("flux2-klein-4b", "black-forest-labs/FLUX.2-klein-4B"),
            pipeline="Flux2KleinPipeline",
            blurb="Compact 4B model. Four-step stills that fit on a consumer GPU.",
            steps=4,
            guidance={"guidance_scale": 1.0},
        ),
        ModelSpec(
            id="wan-2-2",
            name="Wan 2.2",
            kind="video",
            licence="Apache 2.0",
            repo=_repo("wan-2-2", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            pipeline="WanPipeline",
            blurb="Text or start frame to video at 24 fps (5B variant). Cinematic motion.",
            steps=50,
            guidance={"guidance_scale": 5.0},
            supports_negative=True,
            fps=24,
            frame_step=4,
            max_seconds=5,
            i2v_pipeline="WanImageToVideoPipeline",
            # The 5B text+image model conditions on the first frame through expanded timesteps.
            i2v_load={"expand_timesteps": True},
        ),
        ModelSpec(
            id="ltx-2",
            name="LTX-2",
            kind="video",
            licence="LTX-2 Community License",
            repo=_repo("ltx-2", "Lightricks/LTX-2"),
            pipeline="LTX2Pipeline",
            blurb="Text or start frame to video, with synchronised audio in one pass.",
            steps=40,
            guidance={"guidance_scale": 4.0},
            supports_negative=True,
            fps=24,
            frame_step=8,
            max_seconds=10,
            has_audio=True,
            i2v_pipeline="LTX2ImageToVideoPipeline",
            default_negative=_LTX_NEGATIVE,
        ),
        ModelSpec(
            # github.com/Lightricks/LTX-Video, the original v0.9.x model, through its diffusers pipelines.
            id="ltx-video",
            name="LTX-Video",
            kind="video",
            licence="Weights licensed separately",
            repo=_repo("ltx-video", "Lightricks/LTX-Video"),
            pipeline="LTXPipeline",
            blurb="The original LTX model (v0.9.x). Light and fast, silent clips up to 10 s. Retired by its maker.",
            steps=50,
            guidance={"guidance_scale": 3.0},
            supports_negative=True,
            fps=24,
            frame_step=8,
            max_seconds=10,
            i2v_pipeline="LTXImageToVideoPipeline",
            default_negative=_LTX_NEGATIVE,
            # Works best below 720 x 1280; sides must be multiples of 32.
            sizes={"16:9": (1024, 576), "1:1": (704, 704), "9:16": (576, 1024)},
        ),
    ]
}

# (width, height) per aspect ratio. Video sizes are multiples of 32, which both
# video models require; image sizes are multiples of 16.
ASPECTS: dict[str, dict[str, tuple[int, int]]] = {
    "image": {
        "16:9": (1280, 720),
        "1:1": (1024, 1024),
        "9:16": (720, 1280),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    },
    "video": {
        "16:9": (1280, 704),
        "1:1": (768, 768),
        "9:16": (704, 1280),
    },
}
