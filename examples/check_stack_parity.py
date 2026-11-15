"""Verify a fixed two-input parity handoff; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.classifier import MiniRocketClassifier
from ai_server.services.sequence_model import file_hash
from research.minirocket_stack_training import StackMiniRocketClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path, help="Existing collected inference result directory")
    p.add_argument("output", type=Path, help="New local evidence directory")
    a = p.parse_args()
    lock = json.loads((a.evidence / "input_lock.json").read_bytes())
    models = lock["c"]["weights"]
    models["baseline"]["path"] = str(Path("models/minirocket_classifier.joblib").resolve())
    cases = []
    for label in ("bird", "drone"):
        for job in lock["jobs"]:
            if job["label"] != label:
                continue
            preds = {m: json.loads((a.evidence / "predictions" / m / (job["id"] + ".json")).read_bytes())
                     for m in models}
            if all(p["outcome"] in ("correct", "wrong") for p in preds.values()):
                cases.append(dict(job=job, predictions=preds))
                break
    assert len(cases) == 2
    for model in models.values():
        assert file_hash(Path(model["path"])) == model["sha256"]
    for case in cases:
        job = case["job"]
        assert file_hash(Path(job["path"])) == job["sha256"]
        for name, pred in case["predictions"].items():
            assert pred["input_sha256"] == job["sha256"] and pred["model_sha256"] == models[name]["sha256"]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "lock.json").write_text(json.dumps(dict(cases=cases, models=models, tolerance=1e-8), indent=2))
    root = Path(__file__).resolve().parents[1]
    for module in ("ai_server.schemas", "ai_server.services.classifier", "research.minirocket_stack_training",
                   "research.trajectory_sequence", "research.minirocket_inference"):
        from importlib import import_module
        assert Path(import_module(module).__file__).resolve().is_relative_to(root)
    results = []
    for name, adapter in (("baseline", MiniRocketClassifier), ("stack", StackMiniRocketClassifier)):
        model = adapter(models[name]["path"])
        for case in cases:
            job, pred = case["job"], case["predictions"][name]
            result = model.predict(TrackSequence.model_validate_json(Path(job["path"]).read_bytes()))
            assert result.label == pred["label"] and result.window_starts_s == pred["window_starts_s"]
            assert result.window_rejections == pred["window_rejections"]
            np.testing.assert_allclose(result.window_scores, pred["window_scores"], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(result.decision_score, pred["decision_score"], rtol=1e-8, atol=1e-8)
            results.append(dict(model=name, id=job["id"], label=result.label, margin=result.decision_score,
                                delta=abs(result.decision_score-pred["decision_score"]),
                                max_window_delta=max(abs(x-y) for x,y in zip(result.window_scores, pred["window_scores"]))))
    for model in models.values():
        assert file_hash(Path(model["path"])) == model["sha256"]
    (a.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
