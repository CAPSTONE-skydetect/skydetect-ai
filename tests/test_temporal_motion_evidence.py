import numpy as np

from ai_server.services.foreground_motion import TemporalMotionEvidence


def test_history_never_creates_evidence_on_a_blank_current_frame() -> None:
    evidence = TemporalMotionEvidence(0.5)
    first = np.full((20, 20), 40, dtype=np.uint8)
    evidence.update(first, (10, 10), 1)
    assert not evidence.update(np.zeros_like(first), (10, 10), 2).any()


def test_temporal_motion_aligns_the_previous_object_to_the_new_position() -> None:
    evidence = TemporalMotionEvidence(0.5)
    first = np.zeros((20, 20), dtype=np.uint8)
    first[9:12, 9:12] = 40
    evidence.update(first, (10, 10), 1)
    current = np.zeros_like(first)
    current[9:12, 11:14] = 20
    result = evidence.update(current, (12, 10), 2)
    assert result[10, 12] == 30
    assert result[10, 9] == 0


def test_temporal_motion_resets_after_a_frame_gap() -> None:
    evidence = TemporalMotionEvidence(0.5)
    first = np.full((20, 20), 40, dtype=np.uint8)
    evidence.update(first, (10, 10), 1)
    current = np.full_like(first, 20)
    np.testing.assert_array_equal(evidence.update(current, (10, 10), 3), current)
