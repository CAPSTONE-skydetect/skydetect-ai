import json

from research.ard100_followup import compare


def test_failed_candidate_keeps_requested_sample_and_cannot_pass_gate(tmp_path):
    baseline, output = tmp_path / "baseline", tmp_path / "candidate"
    folder = baseline / "samples/clip"
    folder.mkdir(parents=True)
    output.mkdir()
    sample = dict(sample_id="clip", status="ready")
    (output / "protocol.json").write_text(json.dumps(dict(samples=[sample], variant="AR1", development_scope="dev12")))
    report = dict(localization=dict(normalized_threshold_success=dict(within_0_5_bbox_diagonal=dict(full_gt_ratio=.8))),
                  coverage=dict(full_gt_observation_ratio=1.), reference_metrics=dict(error_px=dict(median=2.)))
    (folder / "result.json").write_text(json.dumps(dict(status="completed", tracking_evaluation=report)))
    compare(baseline, output)
    result = json.loads((output / "tracking_comparison.json").read_text())
    assert result["summaries"][1]["requested"] == 1
    assert result["summaries"][1]["mean_correct_localization_ratio"] == 0
    assert result["summaries"][1]["median_error_px_1280"] is None
    assert result["passes_all"] is False
