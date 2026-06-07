import json
from pathlib import Path

from ai_server.schemas import TrackSequence


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "ai_server"
    / "docs"
    / "yolomg_handoff"
    / "drone_fix_stab_01"
)


def test_yolomg_handoff_tracks_match_shared_schema() -> None:
    payload = json.loads((FIXTURE_DIR / "tracks_all.json").read_text())
    tracks = [TrackSequence.model_validate(track) for track in payload["tracks"]]

    assert len(tracks) == 4
    assert [track.track_id for track in tracks] == [1, 2, 3, 4]


def test_yolomg_handoff_selected_track_is_dense_drone_candidate() -> None:
    selected = TrackSequence.model_validate_json(
        (FIXTURE_DIR / "selected_track.json").read_text()
    )
    report = json.loads((FIXTURE_DIR / "selection_report.json").read_text())

    assert selected.track_id == report["selected_track_id"] == 1
    assert selected.quality is not None
    assert selected.quality.num_points == 246
    assert selected.quality.missing_ratio == 0.0
    assert selected.quality.mean_conf == 0.814
    assert report["ground_truth_label"] == "drone"
