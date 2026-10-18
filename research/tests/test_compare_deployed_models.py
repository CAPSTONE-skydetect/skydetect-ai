from research.compare_deployed_models import normalize_label, summarize


def _row(sample, truth, variant, model, prediction, *, confidence=None, margin=None):
    return {
        "sample_id": sample,
        "truth": truth,
        "variant": variant,
        "model": model,
        "prediction": prediction,
        "correct": prediction == truth,
        "confidence": confidence,
        "margin": margin,
    }


def test_normalize_label_collapses_cvat_instance_names():
    assert normalize_label("bird_1") == "bird"
    assert normalize_label("bird_2") == "bird"
    assert normalize_label("drone") == "drone"


def test_summarize_reports_accuracy_and_gt_to_a_transition():
    rows = []
    for variant, rf_prediction, mr_prediction in (
        ("a", "bird", "drone"),
        ("gt_matched", "drone", "drone"),
        ("gt_full", "drone", "drone"),
    ):
        rows.append(
            _row(
                "sample",
                "drone",
                variant,
                "rf",
                rf_prediction,
                confidence=0.8,
            )
        )
        rows.append(
            _row(
                "sample",
                "drone",
                variant,
                "minirocket",
                mr_prediction,
                margin=0.4,
            )
        )

    result = summarize(rows)

    assert result["by_condition"]["rf:a"]["accuracy"] == 0.0
    assert result["by_condition"]["minirocket:a"]["accuracy"] == 1.0
    rf_transition = result["gt_matched_to_a"]["rf"][0]
    assert rf_transition["prediction_changed"] is True
    assert rf_transition["gt_correct_a_wrong"] is True
