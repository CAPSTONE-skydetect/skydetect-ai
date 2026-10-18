import numpy as np
import pytest

from research.ard100_yolo_hybrid_downstream import (
    accepted_recoveries,
    extend_track_payload,
    transform_box,
    transform_point,
)


def test_transform_point_applies_translation_and_scale():
    matrix = np.asarray([[2, 0, 3], [0, 2, -1], [0, 0, 1]], dtype=float)
    assert transform_point(matrix, 4, 5) == (11.0, 9.0)


def test_transform_box_keeps_axis_aligned_bounds_after_rotation():
    matrix = np.asarray([[0, -1, 100], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert transform_box(matrix, 2, 3, 4, 5) == (92.0, 2.0, 5.0, 4.0)


def test_extend_track_only_adds_verified_gap_recovery_and_preserves_d0():
    d0 = {
        "track_id": 1, "processed_width": 1280, "processed_height": 720,
        "stabilization": {"applied": True, "method": "opencv_feature_cmc"},
        "history": [{"frame_index": 0, "timestamp_ms": 0, "cx": .5, "cy": .5,
                     "w": .01, "h": .01, "conf": .9}],
    }
    rows = [
        {"frame_index": "1", "timestamp_ms": "33", "tracking_source": "klt",
         "detector_accepted": "False", "visible": "True"},
        {"frame_index": "2", "timestamp_ms": "67", "tracking_source": "yolo_hybrid_gap_reacquisition",
         "detector_accepted": "True", "visible": "True", "raw_x": "642", "raw_y": "357",
         "bbox_x": "640", "bbox_y": "355", "bbox_width": "4", "bbox_height": "4",
         "confidence": ".7"},
    ]
    matrices = {2: np.asarray([[1, 0, 2], [0, 1, 3], [0, 0, 1]], dtype=float)}
    result = extend_track_payload(d0, rows, matrices)
    assert result["history"][0] == d0["history"][0]
    assert result["history"][1]["frame_index"] == 2
    assert result["history"][1]["cx"] == pytest.approx(644 / 1280)
    assert result["history"][1]["cy"] == pytest.approx(360 / 720)
    assert result["quality"] is None
    assert len(accepted_recoveries(rows)) == 1


def test_extend_track_rejects_a_recovery_without_matrix():
    d0 = {"processed_width": 100, "processed_height": 100,
          "history": [{"frame_index": 0}]}
    row = {"frame_index": "1", "tracking_source": "yolo_hybrid_gap_reacquisition",
           "detector_accepted": "True", "visible": "True", "timestamp_ms": "33",
           "raw_x": "50", "raw_y": "50", "bbox_x": "48", "bbox_y": "48",
           "bbox_width": "4", "bbox_height": "4", "confidence": ".8"}
    with pytest.raises(ValueError, match="no verified CMC matrix"):
        extend_track_payload(d0, [row], {})
