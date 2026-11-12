# 🎬 Yardi Movies Director

A free, self-hosted AI studio for movies, videos, pictures, music and
voiceovers. Describe a scene, a song or a script, call "Action!", collect your
takes on the reel, then cut them into one finished movie.

**No credits are needed for anything.** There are no accounts, no
subscriptions, no paywalled models and no credit meter anywhere in the code,
and the app never calls a paid service. Everything runs on **your own
hardware** with open-weights models, so the only limits are your machine and
your electricity.

## What you can do

- **Text to image** and **text to video** with any of the picture and video models below.
- **Music**: describe a style and get a track of up to 2 minutes, with
  optional sung lyrics, from ACE-Step.
- **Voiceovers**: type a script, pick one of six voices and a pace, and Kokoro
  narrates it. This one needs no GPU.
- **Movies**: press "Add to movie" on stills and clips, choose a voiceover and
  a soundtrack, set the music volume and press "Cut movie". You get one MP4
  with the takes in order and the sound mixed in. Stills hold for 3 seconds.
- **Animate a picture**: press "Animate" on any image take, or upload your own
  picture, and Wan 2.2, LTX-2 or LTX-Video uses it as the first frame of the clip.
- **Arena battles**: tick "Arena battle" in Video or Image mode and two of your
  models, picked at random, make the same scene with the same seed. Their
  names stay hidden until you vote A, B or tie, and your votes build your own
  leaderboard. See [Arena](#arena).
- **Direct the shot**: camera-direction chips and a "Look" preset are added to
  your scene description.
- **Work the reel**: filter by video or image, open stills full size, download,
  reuse a scene with its seed, cancel queued takes, delete finished ones.
- Ctrl + Enter in the scene box calls "Action!".

## Models

| Use   | Model            | Licence (as listed) | Default weights                    |
|-------|------------------|---------------------|------------------------------------|
| Image | Z-Image Turbo    | Apache 2.0          | `Tongyi-MAI/Z-Image-Turbo`         |
| Image | Qwen-Image-2512  | Apache 2.0          | `Qwen/Qwen-Image-2512`             |
| Image | FLUX.2 klein 4B  | Apache 2.0          | `black-forest-labs/FLUX.2-klein-4B`|
| Video | Wan 2.2          | Apache 2.0          | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`  |
| Video | LTX-2 (with audio)| LTX-2 Community License | `Lightricks/LTX-2`            |
| Video | LTX-Video (v0.9.x)| Code Apache 2.0; weights licensed separately | `Lightricks/LTX-Video` |
| Music | ACE-Step 1.5      | Open weights; see model card | `ACE-Step/acestep-v15-xl-turbo-diffusers` |
| Voice | Kokoro            | Apache 2.0          | `hexgrad/Kokoro-82M`               |

> **Check each licence on the model card before commercial use.** The Apache
> 2.0 entries are the ones this project was specified with and were not
> re-verified against the model cards. The two Lightricks entries follow the
> [LTX-Video README](https://github.com/Lightricks/LTX-Video#license): the
> repository's code is Apache 2.0, the model weights are licensed separately,
> and LTX-2's weights are under the LTX-2 Community License. The app itself
> charges nothing for any of them.

### About LTX-Video

[LTX-Video](https://github.com/Lightricks/LTX-Video) is Lightricks' original
v0.9.x model. Its maker has retired it in favour of LTX-2.5 and no longer
develops the repository, but it remains a light, fast option. It is integrated
the way its README recommends, through the diffusers `LTXPipeline` (text to
video) and `LTXImageToVideoPipeline` (start frame to video), at 24 fps, up to
10 seconds, at 1024x576, 704x704 or 576x1024. It produces no audio.

Any repo id can be swapped for a local path or mirror with an environment
variable, e.g. `YARDI_REPO_LTX_2=/models/ltx-2`
(`YARDI_REPO_Z_IMAGE_TURBO`, `YARDI_REPO_QWEN_IMAGE_2512`,
`YARDI_REPO_FLUX2_KLEIN_4B`, `YARDI_REPO_WAN_2_2`, `YARDI_REPO_LTX_VIDEO`,
`YARDI_REPO_ACE_STEP`).
Point `YARDI_REPO_LTX_VIDEO` at a newer diffusers-format LTX-Video checkpoint
to use that instead of the default.

## Arena

The arena is a blind test between the models installed on your machine, in the
style of public model arenas such as [Video Arena](https://arena.ai/video). It
is built into this app and does not connect to that service, which needs a
login, limits free generations per day and offers no free public API. Here
there is no limit on battles.

- Both takes use the same prompt, seed, frame and length (the shorter of the
  two models' limits), so the comparison is like for like.
- Model names are withheld by the server, not just hidden on the page, until
  the battle is voted or discarded.
- Ratings are Elo (start 1000, K = 32), replayed from your stored votes in
  `battles.json`. One vote per battle.
- Votes on placeholder takes are practice rounds and never reach the
  leaderboard, since placeholders say nothing about the models.

## Run it

**Try the interface (any machine, no GPU):**

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:7860
```

Without a GPU stack the app starts its **preview engine**: pictures, video and
music takes are clearly labelled placeholders, not model output. It exists so
the studio can be explored and tested anywhere. Cutting movies is real in
every mode, since it only needs ffmpeg (a copy is installed with
`requirements.txt` if your machine has none).

**Real voiceovers (any machine, no GPU):**

```bash
pip install -r requirements-voice.txt
python app.py
```

Until this is installed, voice takes are placeholder tones that mark timing
only. The offered voices are English; the voice weights (a few hundred MB)
download from Hugging Face on first use.

**Generate for real (NVIDIA GPU):**

```bash
# install the torch build for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements-gpu.txt
python app.py
```

Weights download from Hugging Face on first use of each model (tens of GB in
total). One model is held in memory at a time and swapped on demand. Rough
VRAM guide: the image models and Wan 2.2 5B are aimed at 12-24 GB cards with
CPU offload on; LTX-2 wants more.

**Docker (preview mode):** `docker build -t yardi . && docker run -p 7860:7860 -v yardi-data:/data yardi`

## Settings

| Variable            | Default       | Meaning                                              |
|---------------------|---------------|------------------------------------------------------|
| `YARDI_ENGINE`      | `auto`        | `auto`, `diffusers` or `preview`                     |
| `YARDI_HOST`        | `127.0.0.1`   | Bind address                                         |
| `YARDI_PORT`        | `7860`        | Port                                                 |
| `YARDI_OUTPUT_DIR`  | `./outputs`   | Where takes and `library.json` are stored            |
| `YARDI_CPU_OFFLOAD` | `1`           | `0` keeps the whole pipeline on the GPU (more VRAM)  |
| `YARDI_MAX_QUEUE`   | `200`         | Takes allowed to wait at once (protects the machine; not a usage cap) |
| `YARDI_FFMPEG`      | auto          | Path to a specific ffmpeg binary for movie cuts      |

The app has no login. It binds to localhost by default; put it behind your own
authentication before exposing it to a network.

## API

| Method | Path               | Purpose                                             |
|--------|--------------------|-----------------------------------------------------|
| GET    | `/api/config`      | Models, aspect ratios, active engine                |
| POST   | `/api/generate`    | `{model, prompt, aspect, seconds, count, seed, negative_prompt, start_take}`; music adds `lyrics`, voice adds `voice` and `speed` |
| POST   | `/api/battle`      | `{kind: "video" or "image", prompt, aspect, seconds, seed, start_take}` → two blind takes |
| GET    | `/api/arena`       | Battles, newest first, and the leaderboard            |
| POST   | `/api/battles/<id>/vote` | `{winner: "a", "b" or "tie"}`; reveals the models |
| DELETE | `/api/battles/<id>` | Discard an unvoted battle; its takes stay            |
| POST   | `/api/movie`       | `{clips: [take ids], voice, music, music_volume, still_seconds, title}` → one MP4 |
| POST   | `/api/upload`      | Multipart `image`: adds a picture to the reel as a start frame (25 MB max) |
| GET    | `/api/jobs`        | Every take, newest first                            |
| GET    | `/api/jobs/<id>`   | One take with status and progress                   |
| DELETE | `/api/jobs/<id>`   | Remove a take and its file                          |

## Tests

```bash
pip install pytest && pytest -q
```

The suite drives the full request → queue → render → library flow on the
preview engine, and cuts real movies with ffmpeg. The model paths have not
been run, since they need the weights (and, except for Kokoro, a GPU): the
picture, video and music pipelines follow the signatures in diffusers 0.42,
and the voice path follows the `kokoro` 0.7 package.
