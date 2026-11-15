import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ai_server.services.manual_roi_tracker import _combine_motion_support
from ai_server.services.tracker import execute_manual_tracking
from ai_server.tracking_schemas import ManualTrackingRequest, TrackingTuning


def test_default_and_unknown_profile():
    assert TrackingTuning().tracker_variant == "d0"
    with pytest.raises(ValidationError):
        TrackingTuning(tracker_variant="typo")


def test_disagreement_handoff_and_legacy_blend():
    klt = np.array([20., 10.], dtype=np.float32)
    motion = np.array([10., 10.], dtype=np.float32)
    np.testing.assert_array_equal(
        _combine_motion_support(klt, motion, (20., 20.), handoff_enabled=False),
        .62 * klt + .38 * motion,
    )
    np.testing.assert_array_equal(
        _combine_motion_support(klt, motion, (20., 20.), handoff_enabled=True), motion,
    )


@pytest.mark.parametrize("profile", ["d0", "central_v1", "central_handoff_v1"])
def test_profile_from_request_reaches_tracker_and_metadata(tmp_path, profile):
    spec = importlib.util.spec_from_file_location(
        "video_fixture", Path(__file__).with_name("test_manual_roi_tracking.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    video = tmp_path / "target.mp4"
    fixture._make_twenty_pixel_video(video)
    result = execute_manual_tracking(ManualTrackingRequest(
        source_video_id="profile-smoke", video_path=str(video),
        target_bbox=(40., 54., 32., 32.), stabilize=False,
        resize_width=None, write_overlay=False,
        tuning=TrackingTuning(tracker_variant=profile),
    ), output_dir=tmp_path / profile)
    assert result.metadata["tracker_variant"] == profile
    assert len(result.track.history) >= 40
    assert bool(result.metadata.get("experimental_motion_handoff")) == (profile == "central_handoff_v1")
