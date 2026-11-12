import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from engines import PreviewEngine  # noqa: E402
from registry import MODELS  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app = create_app(out_dir=str(tmp_path), engine=PreviewEngine())
    return app.test_client()


def wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_config_lists_all_free_models(client):
    cfg = client.get("/api/config").get_json()
    assert cfg["name"] == "Yardi Movies Director"
    assert [m["name"] for m in cfg["models"]] == [
        "Z-Image Turbo", "Qwen-Image-2512", "FLUX.2 klein 4B", "Wan 2.2", "LTX-2", "LTX-Video",
    ]
    assert cfg["plan"] == {"price": 0, "credits": "unlimited", "subscription": None}


def test_image_generation_end_to_end(client):
    res = client.post("/api/generate", json={"model": "z-image-turbo", "prompt": "a red kite", "count": 2, "seed": 7})
    assert res.status_code == 202
    jobs = res.get_json()["jobs"]
    assert [j["seed"] for j in jobs] == [7, 8]
    done = wait(client, jobs[0]["id"])
    assert done["status"] == "done" and done["media"] == "image"
    media = client.get(f"/media/{done['file']}")
    assert media.status_code == 200 and media.data[:4] == b"\x89PNG"


def test_video_generation_clamps_length(client):
    res = client.post("/api/generate", json={"model": "wan-2-2", "prompt": "waves", "seconds": 99, "aspect": "9:16"})
    job = res.get_json()["jobs"][0]
    assert job["seconds"] == MODELS["wan-2-2"].max_seconds
    assert (job["width"], job["height"]) == (704, 1280)
    done = wait(client, job["id"])
    assert done["status"] == "done", done
    assert client.get(f"/media/{done['file']}").data[:4] == b"RIFF"


def test_many_generations_are_never_metered(client):
    for _ in range(30):
        assert client.post("/api/generate", json={"model": "flux2-klein-4b", "prompt": "x"}).status_code == 202


@pytest.mark.parametrize("body", [
    {"model": "nope", "prompt": "x"},
    {"model": "ltx-2", "prompt": "  "},
    {"model": "ltx-2", "prompt": "x", "aspect": "4:3"},
    {"model": "ltx-2", "prompt": "x", "seed": "abc"},
    {"model": "ltx-2", "prompt": "x" * 2001},
])
def test_bad_requests_are_rejected(client, body):
    assert client.post("/api/generate", json=body).status_code == 400


def test_delete_removes_take_and_file(client, tmp_path):
    job = client.post("/api/generate", json={"model": "z-image-turbo", "prompt": "x"}).get_json()["jobs"][0]
    done = wait(client, job["id"])
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert not (tmp_path / done["file"]).exists()


def test_library_survives_restart(tmp_path):
    first = create_app(out_dir=str(tmp_path), engine=PreviewEngine()).test_client()
    job = first.post("/api/generate", json={"model": "z-image-turbo", "prompt": "x"}).get_json()["jobs"][0]
    wait(first, job["id"])
    second = create_app(out_dir=str(tmp_path), engine=PreviewEngine()).test_client()
    assert [j["id"] for j in second.get("/api/jobs").get_json()["jobs"]] == [job["id"]]


def test_frame_counts_fit_model_constraints():
    assert MODELS["wan-2-2"].frames_for(5) == 121      # 4k + 1
    assert MODELS["ltx-2"].frames_for(5) == 121        # 8k + 1
    assert (MODELS["ltx-2"].frames_for(3) - 1) % 8 == 0


def png_bytes(size=(900, 500)):
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(buf, "PNG")
    buf.seek(0)
    return buf


def test_upload_then_animate_from_start_frame(client):
    up = client.post("/api/upload", data={"image": (png_bytes(), "frame.png")}, content_type="multipart/form-data")
    assert up.status_code == 201
    frame = up.get_json()
    assert frame["media"] == "image" and frame["uploaded"] and frame["prompt"] == "frame.png"
    for model in ("wan-2-2", "ltx-2", "ltx-video"):
        job = client.post("/api/generate", json={"model": model, "prompt": "push in", "start_take": frame["id"]}).get_json()["jobs"][0]
        assert job["start_take"] == frame["id"]
        done = wait(client, job["id"])
        assert done["status"] == "done", done


def test_generated_image_can_be_a_start_frame(client):
    still = client.post("/api/generate", json={"model": "z-image-turbo", "prompt": "a harbour"}).get_json()["jobs"][0]
    wait(client, still["id"])
    res = client.post("/api/generate", json={"model": "wan-2-2", "prompt": "boats drift", "start_take": still["id"]})
    assert wait(client, res.get_json()["jobs"][0]["id"])["status"] == "done"


def test_start_frame_rules(client):
    frame = client.post("/api/upload", data={"image": (png_bytes(), "f.png")}, content_type="multipart/form-data").get_json()
    # image models cannot start from a frame; unknown or non-image takes are refused
    assert client.post("/api/generate", json={"model": "z-image-turbo", "prompt": "x", "start_take": frame["id"]}).status_code == 400
    assert client.post("/api/generate", json={"model": "wan-2-2", "prompt": "x", "start_take": "missing"}).status_code == 400
    clip = client.post("/api/generate", json={"model": "wan-2-2", "prompt": "x"}).get_json()["jobs"][0]
    wait(client, clip["id"])
    assert client.post("/api/generate", json={"model": "wan-2-2", "prompt": "x", "start_take": clip["id"]}).status_code == 400


def test_upload_rejects_non_images(client):
    import io

    bad = client.post("/api/upload", data={"image": (io.BytesIO(b"<script>alert(1)</script>"), "x.png")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert client.post("/api/upload", data={}, content_type="multipart/form-data").status_code == 400
    assert client.get("/api/jobs").get_json()["jobs"] == []


def test_config_marks_which_models_take_a_start_frame(client):
    models = {m["id"]: m["supports_start_image"] for m in client.get("/api/config").get_json()["models"]}
    assert models == {"z-image-turbo": False, "qwen-image-2512": False, "flux2-klein-4b": False, "wan-2-2": True, "ltx-2": True, "ltx-video": True}


def test_ltx_video_uses_its_own_sizes_and_frame_rule(client):
    spec = MODELS["ltx-video"]
    assert (spec.pipeline, spec.i2v_pipeline) == ("LTXPipeline", "LTXImageToVideoPipeline")
    for aspect, size in {"16:9": (1024, 576), "1:1": (704, 704), "9:16": (576, 1024)}.items():
        job = client.post("/api/generate", json={"model": "ltx-video", "prompt": "x", "aspect": aspect}).get_json()["jobs"][0]
        assert (job["width"], job["height"]) == size and size[0] % 32 == 0 and size[1] % 32 == 0
    frames = spec.frames_for(10)
    assert (frames - 1) % 8 == 0 and frames <= 257


def test_every_pipeline_class_exists_in_diffusers_source():
    """Guards against typos in class names without needing torch installed."""
    import importlib.util

    found = importlib.util.find_spec("diffusers")
    if found is None:
        pytest.skip("diffusers is not installed")
    init = open(found.origin).read()
    for spec in MODELS.values():
        for name in filter(None, (spec.pipeline, spec.i2v_pipeline)):
            assert f'"{name}"' in init, name
