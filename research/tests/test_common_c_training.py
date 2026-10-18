import json
from pathlib import Path

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from research.common_c_dataset import _sha256
from research.common_c_training import _cohorts, run, validate_manifest


def _manifest(tmp_path: Path) -> Path:
    records = []
    for index, (split, label) in enumerate(
        (split, label) for split in ("train", "validation", "test") for label in ("bird", "drone")
    ):
        track = TrackSequence(
            track_id=index, processed_width=1280, processed_height=720,
            stabilization=StabilizationInfo(applied=True, method="opencv_feature_cmc"),
            history=[TrackPoint(frame_index=i, timestamp_ms=round(i / 30 * 1000),
                                cx=0.2 + i * 0.001, cy=0.3 + index * 0.01,
                                w=0.01, h=0.01, conf=1.0) for i in range(91)],
        )
        path = tmp_path / f"track-{index}.json"
        path.write_text(track.model_dump_json(), encoding="utf-8")
        records.append({"sample_id": str(index), "partition": split, "label": label,
                        "source_group_id": str(index), "review_status": "approved", "domain": "real",
                        "a_track": path.name, "gt_track": path.name,
                        "a_sha256": _sha256(path), "gt_sha256": _sha256(path)})
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    return manifest


def test_common_cohorts_preserve_windows_and_do_not_read_test_coordinates(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest, errors = validate_manifest(path)
    assert errors == []
    # The loader skips test coordinates by default, even after preflight validation.
    for record in manifest["records"]:
        if record["partition"] == "test":
            record["a_track"] = "not-loaded.json"
    cohorts, _ = _cohorts(path, manifest, "a", False)
    assert set(cohorts) == {"train", "validation"}
    assert all(row["X"].shape == (4, 60) and len(row["features"]) == 9
               for rows in cohorts.values() for row in rows)


def test_preflight_blocks_group_leakage(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["records"][2]["source_group_id"] = "0"
    path.write_text(json.dumps(manifest))
    _, errors = validate_manifest(path)
    assert any("group leakage" in error for error in errors)


def test_preflight_blocks_unreviewed_labels_without_training(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["records"][0]["review_status"] = "needs_review"
    path.write_text(json.dumps(manifest))
    result = run(path, tmp_path / "result")
    assert result["status"] == "blocked"
    assert any("human review_status" in error for error in result["errors"])
    assert not (tmp_path / "result/rf.joblib").exists()
