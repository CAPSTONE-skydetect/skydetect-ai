from research.ard100_yolo_hybrid_gate import _candidate_rows, _passes_calibration, _select_threshold


def _metrics(localization, observation=0.9, error=10.0, jitter=1.0):
    return {
        "mean_localization": localization,
        "mean_observation": observation,
        "median_video_median_error_px": error,
        "median_video_jitter_px": jitter,
    }


def test_threshold_selection_requires_all_validation_guardrails():
    baseline = _metrics(0.5)
    candidates = {
        0.3: _metrics(0.55, observation=0.86),
        0.4: _metrics(0.54, error=11.0),
        0.5: _metrics(0.53, jitter=1.1),
        0.6: _metrics(0.52),
    }

    threshold, decision = _select_threshold(candidates, baseline)

    assert threshold == 0.6
    assert decision["selected"] is True


def test_no_eligible_threshold_disables_pointwise_switching():
    threshold, decision = _select_threshold({0.4: _metrics(0.505)}, _metrics(0.5))

    assert threshold == 0.0
    assert decision["selected"] is False


def test_candidate_replaces_only_confirmed_low_confidence_points_and_can_fill_gap():
    rows = [
        {"frame_index": 1, "raw_x": 10.0, "raw_y": 10.0, "confidence": 0.9, "visible": "True"},
        {"frame_index": 2, "raw_x": 12.0, "raw_y": 10.0, "confidence": 0.2, "visible": "True"},
    ]
    anchors = {
        1: {"row": dict(rows[0]), "x": 10.0, "y": 10.0, "gap": False},
        2: {"row": dict(rows[1]), "x": 12.0, "y": 10.0, "gap": False},
        3: {"row": {**rows[1], "frame_index": 3}, "x": 14.0, "y": 10.0, "gap": True},
    }
    detection = {
        "confidence": 0.8, "accepted": True, "cx": 13.0, "cy": 10.0,
        "x": 12.0, "y": 9.0, "width": 2.0, "height": 2.0,
    }
    evidence = {
        "detector_threshold": 0.4,
        "detections": {1: detection, 2: detection, 3: detection},
        "supported": {1, 2, 3},
    }

    result, counts = _candidate_rows(rows, anchors, evidence, threshold=0.5)
    by_frame = {int(row["frame_index"]): row for row in result}

    assert by_frame[1]["raw_x"] == 10.0
    assert by_frame[2]["raw_x"] == 13.0
    assert by_frame[3]["raw_x"] == 13.0
    assert counts == {"detector_accepted": 2, "pointwise_replacements": 1, "gap_recoveries": 1}


def test_gate_rejects_observation_loss_even_when_localization_improves():
    assert not _passes_calibration(
        _metrics(0.6, observation=0.87),
        _metrics(0.5, observation=0.9),
    )
