"""원본 궤적을 촬영 그룹 단위 5조각으로 나누고, 5개 case(train 3 / val 1 / test 1)용 패키지를 만든다.

case k: test = 조각 k, validation = 조각 (k-1) mod 5, train = 나머지 3조각.
증강은 split 을 나눈 뒤 각 split 안의 원본에서만 만들며, 위치 오차·끊김 분포(profile)는
그 case 의 train 조각에서만 추정한다. 변형 생성·창 변환은 build_observation_dataset 를 그대로 쓴다.

    python -m research.build_cv5_dataset \
      --input data/c_part/final_observation_v1/originals \
      --label-map data/c_part/final_observation_v1/label_map.csv \
      --audit data/c_part/final_observation_v1/calibration \
      --output research/output/c_part_cv5_v1
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import build_observation_dataset as base

PIECES = 5


def assign_pieces(records, seed):
    splitter = StratifiedGroupKFold(n_splits=PIECES, shuffle=True, random_state=seed)
    labels = [r["label"] for r in records]
    groups = [r["source_group_id"] for r in records]
    for piece, (_, held) in enumerate(splitter.split(records, labels, groups)):
        for i in held:
            records[i]["piece"] = piece
    for piece in range(PIECES):
        if {r["label"] for r in records if r["piece"] == piece} != set(base.TARGETS):
            raise ValueError(f"piece {piece} lacks a label")


def build(root, label_map, audit, output, seed, location_scale=1.0):
    if output.exists():
        raise FileExistsError(output)
    records, links = base.read_sources(root, label_map)
    clips = base.calibration_clips(audit, records)
    assign_pieces(records, seed)
    for record in records:
        record["window_result"] = base.track_windows(record["track"])
        record["rf"] = base.feature_row(record["track"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".cv5-") as temp:
        stage = Path(temp) / output.name
        stage.mkdir()
        cases = {}
        for k in range(PIECES):
            test_piece, val_piece = k, (k - 1) % PIECES
            case_dir = stage / f"case_{k}"
            case_dir.mkdir()
            roles = {}
            for r in records:
                role = "test" if r["piece"] == test_piece else "validation" if r["piece"] == val_piece else "train"
                roles[r["parent_track_id"]] = role
            members = {s: [dict(r, split=s, fold=r["piece"]) for r in records if roles[r["parent_track_id"]] == s]
                       for s in base.SPLITS}
            profile = base.observation_profile(members["train"], clips)
            if location_scale != 1.0:  # 증강 세기 점검(E12): 측정한 위치 오차를 일정 배율로 키우거나 줄인다
                profile = (profile[0] * location_scale, profile[1], dict(profile[2], location_scale=location_scale))
            packages = {}
            for split, selected in members.items():
                for suffix, calibration in (("real", None), ("augmented", profile)):
                    name = f"{split}_{suffix}"
                    packages[name] = base.write_package(case_dir / name, selected, seed + k, calibration)
                    print(f"case_{k}/{name}: {packages[name]}", flush=True)
            cases[f"case_{k}"] = dict(test_piece=test_piece, validation_piece=val_piece,
                                      train_pieces=[p for p in range(PIECES) if p not in (test_piece, val_piece)],
                                      calibration=profile[2], packages=packages)
        pd.DataFrame([{k: v for k, v in row.items() if k not in {"track", "window_result", "rf"}}
                      for row in records]).to_csv(stage / "sources.csv", index=False, encoding="utf-8-sig")
        manifest = dict(seed=seed, source_root=str(root), cases=cases, location_scale=location_scale,
                        piece_policy="StratifiedGroupKFold(5) over source groups; each piece is test once",
                        augmentation_policy="split first, then 9 variants per original inside each split; "
                        "profile estimated from the case train pieces only",
                        variants_per_track=["original", *base.VARIANTS],
                        sequence_config=base.asdict(base.SEQUENCE), channels=base.CHANNELS)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--location-scale", type=float, default=1.0,
                        help="증강 위치 오차 배율 (기본 1 = 측정값 그대로). 분할·원본은 같고 증강 세기만 바뀐다")
    args = parser.parse_args()
    build(args.input.resolve(), args.label_map.resolve(), args.audit.resolve(), args.output.resolve(), args.seed,
          args.location_scale)
