"""Assemble the completed, source-backed MiniRocket development experiments."""

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import statistics

import numpy as np


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, zip_path: Path):
    output = root / "minirocket-results-20261018"
    output.mkdir(exist_ok=True)
    package = root / "received-20261018/real_reference_comparison_v1"
    manifest = _read(package / "dataset_manifest.json")
    inventory = list(csv.DictReader((package / "source_inventory.csv").open()))
    pilot = _read(root / "pilot_manifest.json")["samples"]
    overlaps = [{"sample_id": sample["sample_id"], "group": row["source_group_id"], "split": row["split"]}
                for sample in pilot for row in inventory if row["video_sha256"] == sample["video_sha256"]]
    train = np.load(package / "train_real_only.npz", allow_pickle=False)
    val = np.load(package / "validation_real.npz", allow_pickle=False)
    train_hashes = {_hash_array(x) for x in train["X"]}
    val_hashes = {_hash_array(x) for x in val["X"]}
    old_windows = list(csv.DictReader((root / "minirocket-focus-20261018/windows.csv").open()))
    new_windows = list(csv.DictReader((root / "minirocket-retrain-20261018/pilot_baseline/windows.csv").open()))
    old_scores = {(r["sample_id"], r["input"], r["window_index"]): float(r["margin"]) for r in old_windows}
    reproduction_error = max(abs(old_scores[(r["sample_id"], r["input"], r["window_index"])] - float(r["margin"])) for r in new_windows)
    old_tracks = _read(root / "minirocket-focus-20261018/summary.json")["tracks"]
    new_tracks = _read(root / "minirocket-retrain-20261018/pilot_baseline/summary.json")["tracks"]
    old_labels = {(r["sample_id"], r["input"]): r["prediction"] for r in old_tracks}
    labels_match = all(old_labels[(r["sample_id"], r["input"])] == r["prediction"] for r in new_tracks)
    audit = {"zip_sha256": _hash(zip_path), "manifest_sha256": _hash(package / "dataset_manifest.json"),
             "matches_deployed_training_manifest": _hash(package / "dataset_manifest.json") == "d5dbdd50a28f4113ca751cf12758f92788bc822f62eef5847f2c73ee54f50f8c",
             "all_manifest_files_match": all(_hash(package / name) == expected for name, expected in manifest["file_hashes"].items()),
             "train_groups": len(set(train["group_id"])), "validation_groups": len(set(val["group_id"])),
             "group_overlap": sorted(set(train["group_id"]) & set(val["group_id"])),
             "exact_cross_split_window_duplicate_count": len(train_hashes & val_hashes),
             "pilot_known_video_hash_overlaps": overlaps,
             "unhashed_inventory_rows_by_split": {split: sum(r["split"] == split and not r["video_sha256"] for r in inventory)
                                                    for split in ("train", "validation", "test")},
             "baseline_reproduction_max_window_margin_difference": reproduction_error,
             "baseline_reproduction_pilot_labels_match": labels_match,
             "baseline_reproduction_is_bitwise_exact": reproduction_error == 0,
             "test_arrays_included": manifest["test_included"]}
    train.close()
    val.close()
    if not audit["all_manifest_files_match"] or audit["group_overlap"] or not labels_match:
        raise ValueError("dataset/reproduction audit failed")
    specs = [
        ("baseline", "minirocket-retrain-20261018", "baseline", "pilot_baseline"),
        ("rotation90", "minirocket-retrain-20261018", "rotation90", "pilot_rotation90"),
        ("orbit_pool", "minirocket-orbit-20261018", "orbit", "pilot_orbit"),
        ("context4", "minirocket-context4-20261018-v2", None, None),
        ("frame_scale", "minirocket-framescale-20261018", "framescale", "pilot_framescale"),
        ("equal_dual_scale", "minirocket-dualscale-20261018", None, "pilot_dualscale"),
        ("oof_stack", "minirocket-stack-20261018", None, "pilot_stack"),
    ]
    rows = []
    for name, folder, member, pilot_folder in specs:
        path = root / folder
        payload = _read(path / "status.json")
        scores = payload["models"][member] if member else payload
        if pilot_folder:
            probe = _read(path / pilot_folder / "summary.json")
            correct = probe["aggregation_correct_counts_out_of_six"]["mean"]
        else:
            predictions = _read(path / "pilot_predictions.json")
            correct = {kind: sum(r["correct"] for r in predictions if r["input"] == kind) for kind in ("GT", "D0", "N10")}
        rows.append({"model": name, "validation_group_macro_f1": scores["validation_group_metrics"]["macro_f1"],
                     "validation_group_accuracy": scores["validation_group_metrics"]["accuracy"],
                     "validation_track_macro_f1": scores["validation_track_metrics"]["macro_f1"],
                     "validation_group_drone_recall": scores["validation_group_metrics"]["recall"]["drone"],
                     "validation_group_bird_recall": scores["validation_group_metrics"]["recall"]["bird"],
                     "pilot_gt_correct": correct["GT"], "pilot_d0_correct": correct["D0"], "pilot_n10_correct": correct["N10"]})
    candidate = _read(root / "minirocket-stack-20261018/pilot_stack/summary.json")
    d0 = [row for row in candidate["tracks"] if row["input"] == "D0"]
    limitations = ["All results are development comparisons, not an unused-source test.",
                   "Validation was inspected in past development and in this experiment sequence.",
                   "Pilot hypotheses were explored on the same four videos/six objects.",
                   "Known hashes have no pilot overlap; 11 source inventory rows lack video hashes and re-encodes/session overlap remain unresolved.",
                   "Stack improves D0 4/6 to 5/6, but bird_2 margin is only about 0.0002 and is fragile.",
                   "N10 does not gain classification accuracy; retain D0 as A reference.",
                   "The experimental contract requires its own adapter and is not deployed automatically."]
    result = {"dataset_audit": audit, "models": rows, "candidate": "oof_stack",
              "candidate_model_path": str((root / "minirocket-stack-20261018/stack.joblib").resolve()),
              "d0_object_predictions": d0, "limitations": limitations}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ["# MiniRocket experiments, 2026-10-18", "", "Development candidate: D0 A + dual-scale MiniRocket + source-group OOF meta classifier.",
             "Original validation group macro-F1 is maintained; pilot D0 improves from 4/6 to 5/6.", "",
             "| Model | Validation group F1 (10 groups) | Validation track F1 (11 tracks) | GT /6 | D0 /6 | N10 /6 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['validation_group_macro_f1']:.4f} | {row['validation_track_macro_f1']:.4f} | {row['pilot_gt_correct']} | {row['pilot_d0_correct']} | {row['pilot_n10_correct']} |")
    lines += ["", "## Evidence", "", f"Baseline reproduction maximum window margin difference: {reproduction_error:.3g}.",
              "Baseline pilot labels match, but margins are not bitwise identical. Keep the observed numerical discrepancy in the audit.",
              "All manifest file hashes match. Train/validation groups: 29/10, no group overlap.",
              "The new member keeps image-width-normalized motion extent that the old per-window scale normalization removes.",
              "The meta head uses two mean margins, 33 real training tracks and 29 groups. Its inputs come from source-group out-of-fold predictions.",
              "Member alphas are frozen from reproduced models; meta C=1 is fixed. OOF rows are for meta fitting, not an unbiased performance estimate.",
              "Validation and pilot labels do not enter fitting. This does not make repeatedly inspected validation/pilot data an independent test.",
              "", "## Remaining limits", ""]
    lines += ["- " + item for item in limitations]
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text)
    header = "<tr><th>Model</th><th>Validation group F1</th><th>Validation track F1</th><th>GT /6</th><th>D0 /6</th><th>N10 /6</th></tr>"
    cells = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
        row["model"], f'{row["validation_group_macro_f1"]:.4f}', f'{row["validation_track_macro_f1"]:.4f}',
        row["pilot_gt_correct"], row["pilot_d0_correct"], row["pilot_n10_correct"])) + "</tr>" for row in rows)
    objects = "".join(f'<tr><td>{escape(row["sample_id"])}</td><td>{row["truth"]}</td><td>{row["prediction"]}</td><td>{row["margin"]:.6f}</td></tr>' for row in d0)
    (output / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>MiniRocket experiment results</title><style>body{max-width:1120px;margin:32px auto;padding:20px;background:#f5f3ed;color:#21332d;font:16px/1.65 Georgia,serif}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}li{margin:8px 0}a{color:#125e49}</style>'
        '<h1>MiniRocket 실험 결과</h1><p>D0 + 학습된 두 표현 결합: 개발용 D0 4/6 → 5/6. Validation 그룹 F1 0.899 유지.</p>'
        '<p>새 미사용 영상에서 확인된 최종 성능은 아닙니다. bird_2 점수는 약 0.0002로 경계에 가깝습니다.</p>'
        '<div class="scroll"><table>' + header + cells + '</table></div><h2>D0 객체별 판정</h2><table><tr><th>Object</th><th>Truth</th><th>Prediction</th><th>Meta margin</th></tr>' + objects + '</table>'
        '<h2>한계</h2><ul>' + ''.join('<li>'+escape(item)+'</li>' for item in limitations) + '</ul>'
        '<p><a href="summary.json">계산 근거 JSON</a> · <a href="../minirocket-stack-20261018/pilot_stack/windows.csv">창별 점수 CSV</a> · <a href="../minirocket-stack-20261018/oof_tracks.csv">학습 OOF 근거</a></p></html>')
    return result


def _hash_array(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root.resolve(), args.zip.resolve())
    print(json.dumps({"models": result["models"], "audit": result["dataset_audit"]}, indent=2))
