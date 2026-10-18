"""Run a frozen baseline or the experimental stacked candidate on one A track."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.branch_root.resolve()))
    import joblib
    from ai_server.schemas import TrackSequence
    from ai_server.services.classifier import MiniRocketClassifier
    version = joblib.load(args.model)["contract_version"]
    if version == "trajectory-sequence-dual-scale-stack-1.0.0":
        from minirocket_stack import StackMiniRocketClassifier
        classifier = StackMiniRocketClassifier(args.model)
        score_type = "logistic_meta_margin_mean"
    elif version == "trajectory-sequence-1.0.1":
        classifier = MiniRocketClassifier(args.model)
        score_type = "ridge_margin_mean"
    else:
        raise ValueError("this CLI supports only the frozen baseline and stacked candidate")
    track = TrackSequence.model_validate_json(args.input.read_text(encoding="utf-8-sig"))
    result = asdict(classifier.predict(track))
    result.update(model_version=version, score_type=score_type, score_is_probability=False,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
