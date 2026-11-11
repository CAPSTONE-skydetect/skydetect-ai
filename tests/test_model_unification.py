import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ai_server.model_config import MODEL_NAME
from ai_server.schemas import TrackSequence
from tests.model_parity_probe import make_track, probe


def assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_same(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_frozen_branch_classification_and_openapi_parity():
    fixture = "rf" if MODEL_NAME == "rf" else "mini"
    expected = json.loads((Path(__file__).parent / "fixtures/model_parity" / f"{fixture}.json").read_text())
    assert_same(probe(), expected)


def test_frontend_manual_route_uses_selected_model(tmp_path, monkeypatch):
    from ai_server.main import create_app
    from ai_server.routers import ui

    source = tmp_path / "test.mp4"
    source.touch()
    track = TrackSequence.model_validate(make_track())
    artifacts = {k: str(tmp_path / k) for k in (
        "track_sequence", "trajectory", "overlay", "metrics", "foreground_mask"
    )}
    result = SimpleNamespace(track=track, metadata={}, metrics={}, artifacts=artifacts)
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(ui, "execute_manual_tracking", lambda *a, **kw: result)
    with TestClient(create_app()) as client:
        response = client.post("/api/tracks/manual", json={
            "source_video_id": "test", "video_path": str(source),
            "target_bbox": [10, 10, 20, 20],
        })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == MODEL_NAME
    assert body["tracks"] == [track.model_dump(mode="json")]
    assert set(body["download_urls"]) == {"source_video", *artifacts}
    assert (body["features"] is None) == (MODEL_NAME == "minirocket")
    fixture = "rf" if MODEL_NAME == "rf" else "mini"
    expected = json.loads((Path(__file__).parent / "fixtures/model_parity" / f"{fixture}.json").read_text())
    body["prediction"].pop("processing_time_ms")
    assert_same(body["prediction"], expected["cases"]["smooth"]["response"])


def test_contract_identity_and_training_compatibility():
    from importlib import import_module
    from ai_server.contracts.common import TrackSequence as SharedTrack
    from ai_server.services import train
    assert TrackSequence is SharedTrack
    selected = import_module(f"ai_server.classifiers.{MODEL_NAME}.train")
    assert train is selected
    # Stack's legacy data-result import remains available without loading a model.
    from ai_server.services.classifier import SequencePrediction
    assert SequencePrediction("bird", -1).label == "bird"


def test_invalid_model_fails_before_startup():
    result = subprocess.run(
        [sys.executable, "-c", "import ai_server.model_config"],
        env={**os.environ, "SKYDETECT_MODEL": "typo"}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "SKYDETECT_MODEL must be" in result.stderr


def test_training_cli_help_does_not_train():
    for module in ("train", "evaluate"):
        result = subprocess.run(
            [sys.executable, "-m", f"ai_server.services.{module}", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--output-path" in result.stdout if module == "train" else "--help" in result.stdout
