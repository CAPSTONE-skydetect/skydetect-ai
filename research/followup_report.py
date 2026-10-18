"""Assemble a source-backed follow-up report from existing experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from html import escape
from pathlib import Path

import cv2
import numpy as np

from research.tracking_evaluation import load_cvat_track


VARIANTS = ("D0", "N1", "N2", "N3", "N4", "E1", "N4_native", "N5")


def _html(markdown: str) -> str:
    lines = markdown.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(set(cell) <= set("-:") for cell in cells):
                    tag = "th" if not table else "td"
                    table.append("<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in cells) + "</tr>")
                index += 1
            blocks.append("<div class='table-wrap'><table>" + "".join(table) + "</table></div>")
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append("<li>" + escape(lines[index][2:]) + "</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if line.startswith("## "):
            blocks.append("<h2>" + escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            blocks.append("<h2>" + escape(line[2:]) + "</h2>")
        elif line:
            blocks.append("<p>" + escape(line) + "</p>")
        index += 1
    return "\n".join(blocks)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(root: Path, variant: str) -> dict:
    payload = read(root / "evaluation" / variant / "summary.json")
    rows = payload["samples"]
    with (root / "evaluation" / variant / "b_features.csv").open(newline="") as handle:
        features = list(csv.DictReader(handle))
    acceleration = [abs(float(row["relative_difference_percent"])) for row in features
                    if row["scope"] == "matched_gt" and row["feature"] == "acceleration_p95"
                    and row["relative_status"] == "ok"]
    return {
        "variant": variant,
        "position_p95_median_px_1280": statistics.median(row["reference_metrics"]["error_px"]["p95"] for row in rows),
        "observation_ratio_macro": statistics.mean(row["coverage"]["attempted_window_observation_ratio"] for row in rows),
        "acceleration_relative_error_median_percent": statistics.median(acceleration) if len(acceleration) == len(rows) else None,
        "b_comparable_objects": len(acceleration), "object_count": len(rows),
        "objects": [{"sample_id": row["sample_id"],
                     "position_p95_px_1280": row["reference_metrics"]["error_px"]["p95"],
                     "temporal_error_change_p95_px_1280": row["temporal"]["error_change_px"]["p95"] * 1280 / row["video"]["source_width"],
                     "coverage": row["coverage"]["attempted_window_observation_ratio"],
                     "b_status": row["b_features"]["status"]} for row in rows],
    }


def _holdout_sheet(holdout: Path, output: Path) -> None:
    report = read(holdout / "evaluation/D0/per_video/drone_3_holdout.json")
    metadata = read(Path(report["inputs"]["metadata_json"]))
    gt = load_cvat_track(report["inputs"]["cvat_xml"], fps=metadata["fps"], track_id=0)
    gt_by_frame = {row.frame_index: row for row in gt.observations}
    debug_path = Path(report["inputs"]["trajectory_csv"]).parent / "debug/observations.csv"
    with debug_path.open(newline="") as handle:
        debug = {int(row["frame_index"]): row for row in csv.DictReader(handle)}
    cap = cv2.VideoCapture(metadata["source_video_path"])
    sheet = np.full((640, 960, 3), 245, dtype=np.uint8)
    try:
        for tile, frame_index in enumerate((37, 60, 90, 120, 150, 174)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"cannot decode contact-sheet frame {frame_index}")
            reference = gt_by_frame[frame_index]
            row = debug.get(frame_index)
            gx, gy = round(reference.cx), round(reference.cy)
            cv2.rectangle(image, (round(gx-reference.width/2), round(gy-reference.height/2)),
                          (round(gx+reference.width/2), round(gy+reference.height/2)), (230, 200, 0), 2)
            cv2.drawMarker(image, (gx, gy), (230, 200, 0), cv2.MARKER_CROSS, 12, 2)
            if row:
                ax = round(float(row["raw_x"]) * metadata["width"] / metadata["processed_width"])
                ay = round(float(row["raw_y"]) * metadata["height"] / metadata["processed_height"])
                cv2.drawMarker(image, (ax, ay), (0, 120, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            crop = cv2.getRectSubPix(image, (150, 150), (float(gx), float(gy)))
            crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_NEAREST)
            label = f"f{frame_index} A={row['visible'] if row else 'missing'}"
            cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)
            y, x = (tile // 3) * 320, (tile % 3) * 320
            sheet[y:y+300, x:x+300] = crop
    finally:
        cap.release()
    cv2.imwrite(str(output), sheet)


def build(pilot: Path, holdout: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summaries = [_summary(pilot, variant) for variant in VARIANTS]
    holdout_summary = read(holdout / "evaluation/D0/summary.json")["samples"][0]
    inventory = read(output / "local_inventory.json")
    production = {}
    parity_count = 0
    for variant in ("D0", "N4", "N5", "holdout"):
        folder = output / f"production_{variant}"
        payload = read(folder / "summary.json")
        raw = read(folder / "raw_results.json")
        parity_count += sum(row.get("details", {}).get("research_parity_verified") is True for row in raw)
        errors = [row for row in raw if row.get("status") == "runtime_error"]
        if errors:
            raise ValueError(f"production inference failed for {variant}: {errors}")
        production[variant] = payload["by_condition"]
    result = {"pilot": summaries, "production": production, "production_mr_parity_count": parity_count,
              "holdout": {"sample_id": holdout_summary["sample_id"],
                          "position_error_px_1280": holdout_summary["reference_metrics"]["error_px"],
                          "coverage": holdout_summary["coverage"]["attempted_window_observation_ratio"],
                          "matched_frames": holdout_summary["counts"]["matched_visible_frames"],
                          "gt_frames": holdout_summary["counts"]["gt_visible_frames"]},
              "data_inventory": inventory["summary"],
              "conclusion": "Retain D0. No new A variant passed the position/coverage regression gate.",
              "blockers": ["At least two additional independently filmed bird groups need reviewed GT.",
                           "C collaborator real training tracks, NPZ, metadata and split/source lineage are unavailable locally or in GitHub release assets.",
                           "An unused 10-30 pixel target regime holdout remains unavailable."],
              "limitations": ["Four videos/six objects are development cases, not a final benchmark.",
                              "B matched-GT statistics use each variant's observed frame set; compare coverage alongside them.",
                              "Native B aggregate is withheld because only five of six objects are comparable.",
                              "drone_3 is a larger-object auxiliary case with clipped/repeated GT boxes; derivative GT needs review."]}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _holdout_sheet(holdout, output / "holdout_contact_sheet.png")
    lines = ["# 후속 실험 / Follow-up 2026-10-18", "", "D0를 유지합니다. 새 후보 중 위치·관측률 회귀 기준을 함께 통과한 설정은 없습니다.", "",
             "모든 위치 오차는 폭 1280px 기준입니다. 가속도 오차는 객체별 상대 오차 절댓값의 중앙값입니다.", "",
             "| 설정 | 위치 P95 중앙값(px) | 관측률 | 가속도 P95 특징 오차 | B 비교 객체 |",
             "|---|---:|---:|---:|---:|"]
    for row in summaries:
        acc = row["acceleration_relative_error_median_percent"]
        value = f"{acc:.2f}%" if acc is not None else "전체 비교 불가"
        lines.append(f"| {row['variant']} | {row['position_p95_median_px_1280']:.3f} | {100*row['observation_ratio_macro']:.2f}% | {value} | {row['b_comparable_objects']}/6 |")
    lines += ["", "## 추가 실험 조건", "",
              "- N4: 대비 중심 보정 벡터를 현재·직전 프레임 0.5 비율로 결합. 재획득/LOST에서 상태 초기화.",
              "- N4_native: N4 + 원본 해상도 추적, CMC 분석 폭 1280px.",
              "- N5: 직전 차영상과 현재 차영상을 예상 표적 이동에 맞춰 정렬·결합. 현재 프레임 증거가 0인 곳은 후보가 되지 않음.",
              "- CMC 재현기는 A metadata의 cmc_analysis_width를 따르도록 수정. E1과 N4_native의 B 평가 재실행 완료.",
              "", "## 보조 holdout drone_3", "",
              f"- GT {result['holdout']['gt_frames']}프레임 중 {result['holdout']['matched_frames']}프레임 관측 ({100*result['holdout']['coverage']:.2f}%).",
              f"- 위치 중앙값 {result['holdout']['position_error_px_1280']['median']:.3f}px, P95 {result['holdout']['position_error_px_1280']['p95']:.3f}px.",
              "- 원본 앞 175프레임을 FFV1로 복사하고 모든 디코딩 픽셀의 SHA-256 일치를 확인. 원본과 XML은 수정하지 않음.",
              "- 큰 객체 조건이며, 화면 경계의 bbox 잘림·동일 좌표 반복 때문에 가속도 GT는 검수 필요.",
              "", "## C 배포 구성 실행", "",
              "| 입력 궤적 | RF 정답 | MiniRocket 정답 |", "|---|---:|---:|"]
    for variant, payload in production.items():
        rf, mr = payload["rf:a"], payload["minirocket:a"]
        lines.append(f"| {variant} | {rf['correct']}/{rf['count']} | {mr['correct']}/{mr['count']} |")
    lines += ["", f"MiniRocket production/research 창별 점수·라벨 일치 검증: {parity_count}개 입력.",
              "배포 RF와 MiniRocket은 학습 출처·표현·보류 정책이 달라 알고리즘만의 우열로 해석하지 않습니다.",
              "", "## 데이터 점검", "",
              "- 드론/4.mp4와 새/1.mp4는 SHA-256이 같은 파일. 폴더를 클래스 정답으로 쓰지 않음.",
              "- drone_1/1.mp4와 1.mpg는 재인코딩 의심 쌍. 4개 시점 축소영상 MAE가 모두 2/255 이하. 그룹 분할 전 확인 필요.",
              "- 공통 C 학습 도구는 원본 그룹 분할, 파일 해시, TrackSequence 스키마, 두 클래스, real 도메인과 검수 상태를 검사.",
              "- 실제 학습은 데이터 부족으로 보류. 테스트 fixture로 RF/MiniRocket 학습과 validation 추론 경로만 확인.",
              "", "## 남은 입력", ""]
    lines.extend(f"- {item}" for item in result["blockers"])
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    import os
    def link(path: Path, label: str) -> str:
        return f'<a href="{escape(os.path.relpath(path, output))}">{escape(label)}</a>'
    links = [link(pilot / f"evaluation/{v}/report.html", f"{v} 상세/비교 영상") for v in ("D0", "N4", "N5")]
    links.append(link(holdout / "evaluation/D0/report.html", "drone_3 상세/비교 영상"))
    (output / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>SkyDetect follow-up 2026-10-18</title><style>body{max-width:1080px;margin:36px auto;padding:20px;background:#f6f4ef;color:#202a2b;font:16px/1.7 Georgia,serif}table{border-collapse:collapse;width:100%;background:#fff}td,th{padding:12px;text-align:right;border-bottom:1px solid #c8d3ce}th:first-child,td:first-child{text-align:left}th{background:#e2ebe5}.table-wrap{overflow:auto}img{width:100%;max-width:960px}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#16604c}h2{margin-top:36px}li{margin:8px 0}</style>'
        + '<h1>SkyDetect 실험 검토</h1><nav>' + ' '.join(links) + '</nav>' + _html(text)
        + '<h2>drone_3 GT / A 확대 확인</h2><p>GT: 청록색 · A: 주황색 · A=False는 미관측 예측 위치</p>'
        + '<img src="holdout_contact_sheet.png" alt="GT and A contact sheet"></html>', encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.pilot, args.holdout, args.output)
    print(json.dumps({"conclusion": result["conclusion"], "parity_inputs": result["production_mr_parity_count"]}, indent=2))


if __name__ == "__main__":
    main()
