"""Present frozen ARD100 results, including unavailable inputs and tracking loss."""

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import median


def table(headers, rows):
    escape = lambda value: html.escape(str(value))
    return "<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def build(root):
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    with (root / "classification.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    lookup = {(r["sample_id"], r["input"], r["model"]): r for r in predictions}
    tracking, pairs = [], []
    for sample in protocol["samples"]:
        path = root / "samples" / sample["sample_id"] / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        report = result.get("tracking_evaluation", {})
        reference = report.get("reference_metrics", {})
        coverage = report.get("coverage", {}).get("attempted_window_observation_ratio")
        full_coverage = report.get("coverage", {}).get("full_gt_observation_ratio")
        tracking.append(dict(sample_id=sample["sample_id"], split=sample["split"],
                             status=result.get("status", sample["status"]),
                             reason=result.get("reason", sample.get("reason", "")),
                             observed_ratio_requested_gt=full_coverage,
                             observed_ratio_attempted_gt=coverage,
                             median_error_px_1280=reference.get("error_px", {}).get("median"),
                             p95_error_px_1280=reference.get("error_px", {}).get("p95"),
                             median_error_bbox_width=reference.get("error_bbox_width", {}).get("median"),
                             bbox_width_1280=sample.get("median_bbox_width_1280"),
                             longest_missing_run_frames=report.get("timeline", {}).get("longest_missing_run_frames"),
                             median_error_change_px_1280=reference.get("error_change_px", {}).get("median")))
    for split in ("test", "train"):
        for model in ("baseline", "candidate"):
            samples = [s for s in protocol["samples"] if s["split"] == split]
            counts = dict(both_correct=0, GT_correct_A_wrong=0, GT_wrong_A_correct=0,
                          both_wrong=0, unpaired_or_uncertain=0)
            for s in samples:
                gt, a = (lookup.get((s["sample_id"], kind, model), {}) for kind in ("GT_matched", "D0_matched"))
                gp, ap = gt.get("prediction"), a.get("prediction")
                if gp not in ("drone", "bird") or ap not in ("drone", "bird"):
                    key = "unpaired_or_uncertain"
                elif gp == "drone" and ap == "drone":
                    key = "both_correct"
                elif gp == "drone":
                    key = "GT_correct_A_wrong"
                elif ap == "drone":
                    key = "GT_wrong_A_correct"
                else:
                    key = "both_wrong"
                counts[key] += 1
            pairs.append(dict(split=split, model=model, **counts))
    with (root / "tracking_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tracking[0]))
        writer.writeheader(); writer.writerows(tracking)
    (root / "paired_outcomes.json").write_text(json.dumps(pairs, indent=2))
    primary = [r for r in summary["summaries"] if r["input"] == "D0"]
    primary_rows = [[r["split"], r["model"], f"{r['correct_drone']}/{r['requested']}",
                     f"{100*r['end_to_end_drone_recall']:.1f}%", r["wrong_bird"], r["uncertain"], r["unavailable"]] for r in primary]
    test_scores = {r["model"]: r for r in primary if r["split"] == "test"}
    test_errors = [r["median_error_px_1280"] for r in tracking if r["split"] == "test" and r["median_error_px_1280"] is not None]
    test_width_errors = [r["median_error_bbox_width"] for r in tracking if r["split"] == "test" and r["median_error_bbox_width"] is not None]
    fixed, regressed = [], []
    for s in protocol["samples"]:
        if s["split"] != "test":
            continue
        old, new = (lookup[(s["sample_id"], "D0", name)]["prediction"] for name in ("baseline", "candidate"))
        if old == "bird" and new == "drone":
            fixed.append(s["sample_id"])
        if old == "drone" and new != "drone":
            regressed.append(s["sample_id"])
    findings = dict(test_baseline_correct=test_scores["baseline"]["correct_drone"],
                    test_candidate_correct=test_scores["candidate"]["correct_drone"],
                    test_requested=test_scores["candidate"]["requested"],
                    test_improved=fixed, test_regressed=regressed,
                    test_tracking_report_count=len(test_errors),
                    median_of_test_video_median_error_px_1280=median(test_errors),
                    median_of_test_video_median_error_bbox_width=median(test_width_errors))
    (root / "findings.json").write_text(json.dumps(findings, indent=2))
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    detail_rows = []
    for r in tracking:
        pred = [lookup.get((r["sample_id"], "D0", name), {}) for name in ("baseline", "candidate")]
        detail_rows.append([r["sample_id"], r["split"], fmt(r["bbox_width_1280"]),
                            fmt(100*r["observed_ratio_requested_gt"]) if r["observed_ratio_requested_gt"] is not None else "—",
                            fmt(r["median_error_px_1280"]), fmt(r["p95_error_px_1280"]),
                            fmt(100*r["median_error_bbox_width"]) if r["median_error_bbox_width"] is not None else "—",
                            pred[0].get("prediction", "unavailable"), pred[1].get("prediction", "unavailable"), r["status"]])
    content = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARD100 · Frozen D0 / MiniRocket</title><style>
body{margin:0;background:#f3f1e9;color:#202d32;font:16px/1.65 Georgia,'Apple SD Gothic Neo',serif}main{max-width:1250px;margin:auto;padding:48px 24px}h1{font-size:38px;line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d9dfdb;text-align:left}th{background:#e1ebe6}section{overflow-x:auto}p{max-width:1000px}.note{background:#ffedbf;padding:18px;border-left:4px solid #b17e13}a{color:#17604b}video{width:100%;max-width:600px}code{word-break:break-all}</style><main>
<p>ARD100 / 2026-10-18 / 고정 모델 평가</p><h1>새 영상에서 GT와 A, 두 MiniRocket 비교</h1>
<p>영상별 첫 유효 정답 bbox로만 초기화하고 최대 10초를 D0로 추적했습니다. 이후 GT를 추적기에 제공하거나 모델을 재학습하지 않았습니다. 공식 test/train 구분을 유지했습니다.</p>
<p class="note">드론만 있는 자료입니다. 아래 비율은 전체 요청 영상 중 드론 판정 비율이며, 새 오탐률이나 종합 분류 정확도가 아닙니다. 추적 실패·입력 거부·분류 보류를 분모에서 제외하지 않았습니다.</p>
<h2>A의 실제 출력에 대한 최종 판정</h2>"""
    content += table(["ARD split", "모델", "드론 정답 / 전체", "드론 재현율", "새 오분류", "보류", "실행·입력 불가"], primary_rows)
    content += f"<p>test에서 드론 정답은 {findings['test_baseline_correct']}개 → {findings['test_candidate_correct']}개입니다. 추가 정답 영상: {', '.join(fixed)}. 기존 정답에서 새 오답으로 바뀐 영상은 {len(regressed)}개입니다. 이 결과만으로 새 오탐률 개선이나 종합 분류 성능 우위를 주장할 수 없습니다.</p>"
    content += f"<p class='note'>추적 정확도는 별도로 주의해야 합니다. test의 평가 가능한 {len(test_errors)}개 영상에서, 영상별 중심 오차 중앙값의 중앙값은 {median(test_errors):.2f}px입니다. 객체 폭 대비 오차 중앙값의 영상 간 중앙값은 {100*median(test_width_errors):.1f}%입니다. 관측률이 높거나 분류가 맞더라도 다른 배경을 따라간 경우가 있습니다. 분류 정답 수를 올바른 표적 추적 성공 수로 해석하면 안 됩니다.</p>"
    content += "<h2>같은 관측 프레임의 GT와 A</h2><p>GT_matched와 D0_matched의 관측 프레임·시각을 맞춘 비교입니다. GT를 넣어도 틀리는 경우와 A에서만 틀리는 경우를 구분합니다.</p>"
    content += table(["Split", "모델", "둘 다 정답", "GT만 정답", "A만 정답", "둘 다 오답", "비교 불가·보류"],
                     [[r[k] for k in ("split", "model", "both_correct", "GT_correct_A_wrong", "GT_wrong_A_correct", "both_wrong", "unpaired_or_uncertain")] for r in pairs])
    content += "<h2>모든 입력 종류</h2><p>GT_full은 A가 실제 시도한 구간의 모든 GT이며, 원본 영상 전체 길이가 아닙니다. D0는 실제 전체 A 출력, D0_matched는 GT가 있는 프레임의 A입니다.</p>"
    content += table(["Split", "입력", "모델", "정답", "오답", "보류", "불가", "전체"],
                     [[r[k] for k in ("split", "input", "model", "correct_drone", "wrong_bird", "uncertain", "unavailable", "requested")] for r in summary["summaries"]])
    content += "<h2>영상별 추적 및 분류</h2><p>좌표 오차는 폭 1280px 기준입니다. 관측률은 요청 구간의 주석 있는 프레임 대비이며, 조기 종료 후 구간도 분모에 남깁니다. 위치 오차는 관측된 프레임에서만 계산되므로 관측률과 함께 읽어야 합니다.</p><section>"
    content += table(["영상", "Split", "GT 폭 px", "관측률 %", "중심 Median px", "P95 px", "폭 대비 Median %", "기존 C", "새 C", "상태"], detail_rows) + "</section>"
    content += "<h2>정합성 확인과 비교 영상</h2>"
    for name in ("phantom02", "phantom04", "phantom141", "phantom09"):
        image = root / f"alignment_{name}.jpg"
        video = root / "samples" / name / "comparison.mp4"
        if image.exists():
            content += f'<p>{name} · 초기/중간 XML 박스</p><img style="width:100%" src="{image.name}">'
        if video.exists():
            content += f'<video controls preload="none" src="samples/{name}/comparison.mp4"></video>'
    if (root / "qa_drift_phantom133.jpg").exists():
        content += '<h2>분류가 맞아도 추적이 틀릴 수 있는 사례</h2><p>phantom133: 초록은 GT, 빨강은 A입니다. 새 C가 드론으로 판정하더라도 A의 중심이 배경 쪽으로 이탈한 프레임이 있습니다.</p><img style="width:100%" src="qa_drift_phantom133.jpg">'
    content += "<h2>해석의 한계</h2><p>출처가 기존 학습 자료와 겹치지 않는다는 사용자 확인을 기록했습니다. ARD100 영상끼리의 촬영 세션 독립성은 보장하지 않습니다. VOC 좌표를 그대로 보존했으며 서브픽셀 수준 정답 정확도를 가정하지 않습니다. CMC 보정 GT가 [0,1]을 벗어나면 해당 입력을 거부하며 조용히 클램프하지 않습니다. 10초 이후의 장기 추적 성능은 이 결과에 포함되지 않습니다.</p>"
    content += '<p>근거: <a href="protocol.json">고정 프로토콜</a> · <a href="classification.csv">분류 CSV</a> · <a href="classification_details.json">창별 점수</a> · <a href="tracking_summary.csv">추적 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="paired_outcomes.json">GT/A 대응</a></p></main></html>'
    (root / "index.html").write_text(content)
    return dict(tracking_reports=sum(r["median_error_px_1280"] is not None for r in tracking), paired_outcomes=pairs, primary=primary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False, indent=2))
