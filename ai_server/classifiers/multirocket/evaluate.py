"""TrackSequence JSON 하나를 MultiRocket Stack 으로 판정해 출력한다.

배포 번들은 데이터 전체로 학습했으므로 데이터셋 점수를 다시 재지 않는다.
보고할 성능은 5-case 교차검증 결과(research/output/c_part_cv5_FINAL_multirocket)다.

    python -m ai_server.services.evaluate --track path/to/track_sequence.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_server.classifiers.multirocket.classifier import DEFAULT_MODEL_PATH, MultiRocketStackClassifier
from ai_server.classifiers.minirocket.contracts import TrackSequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    track = TrackSequence.model_validate_json(args.track.read_bytes())
    result = MultiRocketStackClassifier(args.model).predict(track)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
