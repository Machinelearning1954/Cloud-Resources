"""Audio for Yardi Movies Director: WAV writing, voiceovers and placeholder sound.

Voiceovers use Kokoro, a small open-weights text-to-speech model that runs on
an ordinary CPU. When Kokoro is not installed, a clearly flagged placeholder
tone track is written instead so the rest of the studio still works.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import wave
from array import array
from typing import Callable

PLACEHOLDER_RATE = 22050
VOICE_RATE = 24000        # Kokoro's output sample rate
_voice_pipelines: dict = {}


def write_wav(path: str, pcm: bytes, rate: int, channels: int = 1) -> None:
    """Write 16-bit little-endian PCM to a WAV file."""
    with wave.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)


def write_tensor(path: str, samples, rate: int) -> None:
    """Write a float tensor/array shaped (channels, samples) or (samples,) in [-1, 1]."""
    import numpy as np

    data = samples.detach().float().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples, dtype="float32")
    if data.ndim == 1:
        data = data[None, :]
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    write_wav(path, pcm.T.tobytes(), rate, channels=pcm.shape[0])   # .T interleaves the channels


def _tone(out: array, freq: float, seconds: float, volume: float, rate: int = PLACEHOLDER_RATE) -> None:
    n = int(seconds * rate)
    fade = max(1, int(0.02 * rate))
    for i in range(n):
        envelope = min(1.0, i / fade, (n - i) / fade) * math.exp(-2.5 * i / max(n, 1))
        out.append(int(32767 * volume * envelope * math.sin(2 * math.pi * freq * i / rate)))


def placeholder_music(path: str, seconds: float, seed: int) -> None:
    """A seeded pentatonic arpeggio. Stands in for a music model; it is not model output."""
    rng = random.Random(seed)
    root = 220.0 * 2 ** (rng.randrange(0, 7) / 12)
    scale = [0, 3, 5, 7, 10, 12, 15]
    beat = rng.choice([0.25, 0.3, 0.375])
    out = array("h")
    while len(out) < int(seconds * PLACEHOLDER_RATE):
        _tone(out, root * 2 ** (rng.choice(scale) / 12), beat, 0.35)
    del out[int(seconds * PLACEHOLDER_RATE):]
    write_wav(path, out.tobytes(), PLACEHOLDER_RATE)


def placeholder_voice(path: str, text: str, speed: float) -> None:
    """One short blip per word at roughly speaking pace. Not speech; a timing stand-in."""
    out = array("h")
    for word in text.split()[:400]:
        length = min(0.45, 0.12 + 0.035 * len(word)) / speed
        _tone(out, 150.0 + (sum(map(ord, word)) % 60), length, 0.4)
        out.extend([0] * int(0.06 * PLACEHOLDER_RATE / speed))
    write_wav(path, out.tobytes(), PLACEHOLDER_RATE)


def voice_ready() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("kokoro", "torch", "numpy"))


def speak(path: str, text: str, voice: str, speed: float, progress: Callable[[float], None]) -> None:
    """Synthesize `text` with Kokoro. The first letter of a voice id is its language code."""
    import numpy as np
    from kokoro import KPipeline

    lang = voice[0]
    pipeline = _voice_pipelines.get(lang)
    if pipeline is None:
        pipeline = _voice_pipelines[lang] = KPipeline(lang_code=lang)
    chunks = []
    paragraphs = max(1, text.count("\n") + 1)
    for i, result in enumerate(pipeline(text, voice=voice, speed=speed)):
        if result.audio is not None:
            chunks.append(result.audio.detach().float().cpu().numpy())
        progress(min(0.95, (i + 1) / (paragraphs + 1)))
    if not chunks:
        raise RuntimeError("The voice model produced no audio for this script.")
    write_tensor(path, np.concatenate(chunks), VOICE_RATE)


def voiceover(req: dict, out_dir: str, stem: str, progress: Callable[[float], None]) -> dict:
    name = stem + ".wav"
    path = os.path.join(out_dir, name)
    if voice_ready():
        speak(path, req["prompt"], req["voice"], req["speed"], progress)
        return {"file": name, "media": "audio", "placeholder": False}
    placeholder_voice(path, req["prompt"], req["speed"])
    return {"file": name, "media": "audio", "placeholder": True}


def wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as f:
        return f.getnframes() / float(f.getframerate())
