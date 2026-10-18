import numpy as np

from ai_server.schemas import TrackPoint, TrackSequence
from research.trajectory_refinement import refine_track


def _track(indices, *, jitter=0.0):
    return TrackSequence(
        track_id=1,
        processed_width=1280,
        processed_height=720,
        history=[
            TrackPoint(
                frame_index=index,
                timestamp_ms=index * 40,
                cx=(100 + index * 2 + jitter * (-1) ** index) / 1280,
                cy=200 / 720,
                w=10 / 1280,
                h=10 / 720,
                conf=0.8,
            )
            for index in indices
        ],
    )


def test_kf_preserves_observation_contract_and_reduces_alternating_noise():
    track = _track(range(30), jitter=1.0)
    result = refine_track(track, mode="kf")
    assert [p.frame_index for p in result.track.history] == list(range(30))
    assert [p.timestamp_ms for p in result.track.history] == [i * 40 for i in range(30)]
    raw = np.asarray([p.cx * 1280 for p in track.history])
    filtered = np.asarray([p.cx * 1280 for p in result.track.history])
    assert np.std(np.diff(raw, n=2)) > np.std(np.diff(filtered, n=2))


def test_rts_does_not_fill_gaps_and_splits_long_segments():
    track = _track([0, 1, 2, 20, 21, 22])
    result = refine_track(track, mode="rts", max_segment_gap_seconds=0.25)
    assert result.segment_count == 2
    assert [p.frame_index for p in result.track.history] == [0, 1, 2, 20, 21, 22]


def test_adaptive_measurement_noise_uses_diagnostics_not_confidence_inverse():
    track = _track(range(3))
    rows = {
        0: {"tracking_source": "klt", "forward_backward_error": 0.1},
        1: {"tracking_source": "appearance", "appearance_psr": 1.0},
        2: {"tracking_source": "motion"},
    }
    result = refine_track(
        track, mode="kf", adaptive_measurement=True,
        observation_diagnostics=rows,
    )
    sigmas = [row["measurement_sigma"] for row in result.diagnostics]
    assert sigmas[1] > sigmas[0]
    assert sigmas[2] > sigmas[0]
