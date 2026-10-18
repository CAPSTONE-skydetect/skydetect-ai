import numpy as np
import pytest

from ai_server.services.complementary_center import ComplementaryCenter


def test_constant_offset_anchor_does_not_change_motion():
    tracker = ComplementaryCenter()
    for index in range(80):
        motion = np.array([index * 2.0, index * 0.5])
        actual = tracker.update(motion + [5, -3], motion, index * 40, max_disagreement=32)
        assert np.allclose(actual, motion + [5, -3])


def test_high_frequency_anchor_noise_is_attenuated():
    tracker = ComplementaryCenter()
    errors = []
    for index in range(80):
        motion = np.array([index * 2., 30.])
        anchor = motion + [(-1.)**index, 0]
        output = tracker.update(anchor, motion, index * 40, max_disagreement=32)
        if index > 20:
            errors.append(abs(output[0] - motion[0]))
    assert max(errors) < .15


def test_missing_motion_and_disagreement_fall_back_without_filling_gap():
    tracker = ComplementaryCenter()
    assert np.allclose(tracker.update([10, 10], [0, 0], 0, max_disagreement=4), [10, 10])
    assert np.allclose(tracker.update([12, 12], None, 40, max_disagreement=4), [12, 12])
    assert np.allclose(tracker.update([13, 13], [12, 12], 80, max_disagreement=4), [13, 13])
    with pytest.raises(ValueError):
        tracker.update([0, 0], None, 80, max_disagreement=4)
