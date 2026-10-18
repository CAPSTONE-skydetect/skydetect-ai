import pytest

from ai_server.schemas import TrackSequence
from research.cvat_gt_export import cvat_to_track_sequence, _class_label
from research.tracking_evaluation import CvatTrack, GroundTruthObservation, TrackingEvaluationError


def test_exports_gt_with_original_frame_gaps_and_processed_dimensions() -> None:
    gt = CvatTrack(1, (1,), "bird_1", 1920, 1080, 300, (
        GroundTruthObservation(100, 4000, 960, 540, 24, 12, False, True),
        GroundTruthObservation(102, 4080, 990, 570, 30, 15, False, False),
    ))
    result = cvat_to_track_sequence(gt, fps=25, processed_width=1280,
                                    processed_height=720, source_video_id="GT_source")
    TrackSequence.model_validate_json(result.model_dump_json())
    assert [p.frame_index for p in result.history] == [100, 102]
    assert [p.timestamp_ms for p in result.history] == [4000, 4080]
    assert result.history[0].cx == .5 and result.history[0].cy == .5
    assert result.history[0].w * result.processed_width == 16
    assert result.history[0].conf == 1
    assert result.quality.missing_ratio == pytest.approx(1 / 3)
    assert result.stabilization.applied is False
    assert result.stabilization.method == "none"


def test_rejects_unexplained_aspect_ratio_changes() -> None:
    gt = CvatTrack(0, (0,), "drone", 1920, 1080, 1,
                   (GroundTruthObservation(0, 0, 960, 540, 10, 10, False, True),))
    with pytest.raises(TrackingEvaluationError, match="aspect-preserving"):
        cvat_to_track_sequence(gt, fps=30, processed_width=1280,
                               processed_height=1280, source_video_id="GT_source")


def test_rf_label_comes_from_xml_not_filename() -> None:
    assert _class_label("bird_1") == "bird"
    assert _class_label("bird_2") == "bird"
    assert _class_label("drone") == "drone"
    assert _class_label("unknown") is None
