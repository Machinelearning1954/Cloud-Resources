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


def test_config_lists_five_free_models(client):
    cfg = client.get("/api/config").get_json()
    assert cfg["name"] == "Yardi Movies Director"
    assert [m["name"] for m in cfg["models"]] == [
        "Z-Image Turbo", "Qwen-Image-2512", "FLUX.2 klein 4B", "Wan 2.2", "LTX-2",
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
