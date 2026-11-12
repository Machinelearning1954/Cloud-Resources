"""Cut takes into one finished movie with ffmpeg.

Clips (videos, animated previews and stills) are normalised to one frame size
and frame rate, joined in order, and an optional voiceover and soundtrack are
mixed over whatever sound the clips already carry.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable

FPS = 24
MAX_SIDE = 1280
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def ffmpeg_path() -> str | None:
    """YARDI_FFMPEG, then ffmpeg on PATH, then the binary bundled with imageio-ffmpeg."""
    found = os.environ.get("YARDI_FFMPEG") or shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run(ff: str, args: list[str]) -> None:
    done = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if done.returncode != 0:
        tail = " ".join(done.stderr.strip().splitlines()[-2:]) or "unknown error"
        raise RuntimeError(f"ffmpeg could not finish the cut: {tail}")


def probe(ff: str, path: str) -> tuple[float, bool]:
    """(duration in seconds, has an audio stream), read from ffmpeg's own description of the file."""
    text = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    seconds = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3]) if match else 0.0
    return seconds, bool(re.search(r"Stream #\S+.*Audio:", text))


def frame_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(width, height))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def _animation_frames(path: str, folder: str) -> float:
    """Unpack an animated WebP (ffmpeg cannot read those) into PNGs. Returns its frame rate."""
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        duration = img.info.get("duration") or 83
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            frame.convert("RGB").save(os.path.join(folder, f"f{i:05d}.png"))
    return 1000.0 / duration


def cut(ff: str, out_dir: str, stem: str, job: dict, progress: Callable[[float], None]) -> dict:
    width, height = frame_size(job["width"], job["height"])
    fit = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"
    )
    silence = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    clips = job["clips"]
    steps = len(clips) + 2

    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".cut-") as work:
        segments = []
        for i, clip in enumerate(clips):
            source = os.path.join(out_dir, clip["file"])
            if not os.path.exists(source):
                raise RuntimeError("A take in this movie was deleted before the cut finished.")
            segment = os.path.join(work, f"seg{i:03d}.mp4")
            if clip["media"] == "image":
                picture = ["-loop", "1", "-t", str(job["still_seconds"]), "-i", source]
                _run(ff, [*picture, *silence, "-vf", fit, "-shortest", *VIDEO, *AUDIO, segment])
            elif clip["media"] == "animation":
                frames = os.path.join(work, f"frames{i:03d}")
                os.mkdir(frames)
                rate = _animation_frames(source, frames)
                picture = ["-framerate", f"{rate:.3f}", "-i", os.path.join(frames, "f%05d.png")]
                _run(ff, [*picture, *silence, "-vf", fit, "-shortest", *VIDEO, *AUDIO, segment])
            elif probe(ff, source)[1]:
                _run(ff, ["-i", source, "-vf", fit, *VIDEO, *AUDIO, segment])
            else:
                _run(ff, ["-i", source, *silence, "-vf", fit, "-shortest", *VIDEO, *AUDIO, segment])
            segments.append(segment)
            progress((i + 1) / steps)

        listing = os.path.join(work, "list.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{os.path.basename(s)}'\n" for s in segments)
        joined = os.path.join(work, "joined.mp4")
        _run(ff, ["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", joined])
        length = probe(ff, joined)[0]
        progress((len(clips) + 1) / steps)

        name = stem + ".mp4"
        final = os.path.join(out_dir, name)
        inputs, filters, mix = ["-i", joined], ["[0:a]aresample=48000[base]"], ["[base]"]
        stereo = "aresample=48000,aformat=channel_layouts=stereo"
        note = ""
        if job.get("voice_file"):
            voice = os.path.join(out_dir, job["voice_file"])
            if probe(ff, voice)[0] > length + 0.25:
                note = "The voiceover is longer than the picture, so its end was cut off. Add clips or shorten the script."
            inputs += ["-i", voice]
            filters.append(f"[{len(mix)}:a]{stereo}[voice]")
            mix.append("[voice]")
        if job.get("music_file"):
            # Loop the soundtrack so it always covers the picture; the mix stops with the picture.
            inputs += ["-stream_loop", "-1", "-i", os.path.join(out_dir, job["music_file"])]
            filters.append(f"[{len(mix)}:a]{stereo},volume={job['music_volume']:.2f}[music]")
            mix.append("[music]")
        if len(mix) == 1:
            _run(ff, ["-i", joined, "-c", "copy", "-movflags", "+faststart", final])
        else:
            filters.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0[out]")
            _run(ff, [
                *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[out]",
                "-c:v", "copy", *AUDIO, "-t", f"{length:.3f}", "-movflags", "+faststart", final,
            ])

    result = {"file": name, "media": "video", "seconds": round(length, 1), "width": width, "height": height}
    if note:
        result["note"] = note
    return result
