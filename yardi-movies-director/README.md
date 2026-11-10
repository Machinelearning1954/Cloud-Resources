# 🎬 Yardi Movies Director

A free, self-hosted AI image and video studio. Describe a scene, pick a model,
call "Action!" and collect your takes on the reel.

There are no accounts, no subscriptions, no paywalled models and no credit
meter anywhere in the code. Generations run on **your own hardware** with
open-weights models, so the number of takes is limited only by your GPU.

## Models

| Use   | Model            | Licence (as listed) | Default weights                    |
|-------|------------------|---------------------|------------------------------------|
| Image | Z-Image Turbo    | Apache 2.0          | `Tongyi-MAI/Z-Image-Turbo`         |
| Image | Qwen-Image-2512  | Apache 2.0          | `Qwen/Qwen-Image-2512`             |
| Image | FLUX.2 klein 4B  | Apache 2.0          | `black-forest-labs/FLUX.2-klein-4B`|
| Video | Wan 2.2          | Apache 2.0          | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`  |
| Video | LTX-2 (with audio)| Apache 2.0         | `Lightricks/LTX-2`                 |

> **Check each licence on the model card before commercial use.** The licences
> above are the ones this project was specified with; they were not
> re-verified against the model cards when this was written. LTX-2 in
> particular has been distributed under Lightricks' own community licence,
> which carries terms Apache 2.0 does not.

Any repo id can be swapped for a local path or mirror with an environment
variable, e.g. `YARDI_REPO_LTX_2=/models/ltx-2`
(`YARDI_REPO_Z_IMAGE_TURBO`, `YARDI_REPO_QWEN_IMAGE_2512`,
`YARDI_REPO_FLUX2_KLEIN_4B`, `YARDI_REPO_WAN_2_2`).

## Run it

**Try the interface (any machine, no GPU):**

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:7860
```

Without a GPU stack the app starts its **preview engine**: every take is a
clearly labelled placeholder storyboard, not model output. It exists so the
studio can be explored and tested anywhere.

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

The app has no login. It binds to localhost by default; put it behind your own
authentication before exposing it to a network.

## API

| Method | Path               | Purpose                                             |
|--------|--------------------|-----------------------------------------------------|
| GET    | `/api/config`      | Models, aspect ratios, active engine                |
| POST   | `/api/generate`    | `{model, prompt, aspect, seconds, count, seed, negative_prompt}` |
| GET    | `/api/jobs`        | Every take, newest first                            |
| GET    | `/api/jobs/<id>`   | One take with status and progress                   |
| DELETE | `/api/jobs/<id>`   | Remove a take and its file                          |

## Tests

```bash
pip install pytest && pytest -q
```

The suite drives the full request → queue → render → library flow on the
preview engine. The GPU path (`DiffusersEngine`) follows the pipeline
signatures in diffusers 0.42 but has not been run in CI, since that needs a
GPU and the model weights.
