"""Part C: MiniRocket + Ridge 평가 스크립트.

B 패키지의 세 학습 구성(실제-only / 실제+증강 / 합성-only)을 각각 새로 학습하고
**같은 실제 validation** 에서 창·track·원본 영상 그룹 단위로 비교한다.

    python -m ai_server.services.evaluate                    # reports/minirocket/ 에 저장
    python -m ai_server.services.evaluate --publish          # docs/ 에 확정본 복사
    python -m ai_server.services.evaluate --margin-threshold 0.2   # 보류 정책 결과 추가

주의: validation 은 개발 중 반복 사용된 자료이고 과거 test 는 이미 열람했다.
여기 숫자는 개발 비교이지 새 실제 영상에 대한 최종 성능이 아니다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ai_server.utils.metrics_plot import experiment_card
from ai_server.services.sequence_model import (
    ALPHA_SOURCE,
    ARMS,
    DEFAULT_PACKAGE,
    PROJECT_ROOT,
    REPRESENTATION_FIT,
    aggregate_scores,
    choose_alpha,
    decision_scores,
    file_hash,
    fit_arm,
    group_bootstrap_ci,
    library_versions,
    load_package,
    metrics,
    predict_labels,
)

_DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "minirocket"
_DOCS_DIR = PROJECT_ROOT / "docs"
_LEVELS = ("window", "track", "group")


def evaluate(
    package: Path = DEFAULT_PACKAGE,
    report_dir: Path = _DEFAULT_REPORT_DIR,
    margin_threshold: float = 0.0,
) -> dict:
    package, report_dir = Path(package), Path(report_dir)
    manifest, train, validation = load_package(package)
    report_dir.mkdir(parents=True, exist_ok=True)

    alphas = {}
    for source in sorted(set(ALPHA_SOURCE.values())):
        alphas[source], table = choose_alpha(train[source])
        table.to_csv(report_dir / f"{source}_alpha_cv.csv", index=False)

    results = {}
    for arm in ARMS:
        alpha = alphas[ALPHA_SOURCE[arm]]
        model = fit_arm(train, arm, alpha)
        # 첫 호출은 numba JIT 준비 시간이 섞이므로 준비 후 시간과 따로 잰다.
        start = time.perf_counter()
        decision_scores(model, validation["X"][:1])
        first_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        decisions = decision_scores(model, validation["X"])
        warm_ms = (time.perf_counter() - start) * 1000 / len(validation["X"])

        tables = aggregate_scores(validation, decisions)
        for level, table in zip(_LEVELS, tables):
            table = table.copy()
            table["prediction"] = predict_labels(table.decision, margin_threshold)
            table.to_csv(report_dir / f"{arm}_validation_{level}.csv", index=False)
        results[arm] = dict(
            alpha=alpha,
            representation_fit=REPRESENTATION_FIT[arm],
            train_windows=model["train_windows"],
            train_groups=model["train_groups"],
            validation={level: metrics(table, margin_threshold) for level, table in zip(_LEVELS, tables)},
            group_bootstrap=group_bootstrap_ci(tables[2]),
            errors=_errors(tables[2]),
            latency_ms=dict(first_call=round(first_ms, 1), per_window_warm=round(warm_ms, 3)),
        )
        group = results[arm]["validation"]["group"]
        print(f"{arm}: group macro-F1={group['macro_f1']:.4f} "
              f"(bird recall {group['recall']['bird']:.3f}, drone recall {group['recall']['drone']:.3f})",
              flush=True)

    report = dict(
        status="development_comparison_only",
        package_manifest_sha256=file_hash(package / "dataset_manifest.json"),
        contract_version=manifest["contract_version"],
        contract_id=manifest["contract_id"],
        validation=dict(windows=len(validation["y"]), groups=len(set(validation["group_id"]))),
        validation_previously_reused=manifest["validation_previously_reused"],
        independent_real_world_evaluation_ready=False,
        test_opened=False,
        score_is_probability=False,
        aggregation="window margin mean per track, then track mean per source-video group",
        margin_threshold=margin_threshold,
        versions=library_versions(),
        results=results,
    )
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    plot_card(report, report_dir / "card.png")
    print(f"리포트 저장: {report_dir}")
    return report


_ARM_KO = {"real_only": "실제-only", "real_plus_augmentation": "실제+증강", "synthetic_only": "합성-only"}


def plot_card(report: dict, path: Path, title: str = "MiniRocket + Ridge") -> Path:
    """세 학습 구성을 원본 영상 그룹 단위 카드 한 장으로 그린다."""
    arms = []
    for arm, row in report["results"].items():
        group = row["validation"]["group"]
        boot = row["group_bootstrap"]
        arms.append(dict(
            name=_ARM_KO.get(arm, arm),
            n_label=f"영상 {group['n']}개 · 학습 {row['train_windows']:,}창",
            cm=group["confusion_matrix"],
            accuracy=group["accuracy"], balanced_accuracy=group["balanced_accuracy"],
            macro_f1=group["macro_f1"], roc_auc=group["roc_auc"],
            precision=group["precision"], recall=group["recall"],
            ci=dict(macro_f1=boot["macro_f1_95ci"], balanced_accuracy=boot["balanced_accuracy_95ci"]),
            coverage=group.get("abstain", {}).get("coverage"),
        ))
    subtitle = (f"실제 A validation · 원본 영상 그룹 단위 · 창 {report['validation']['windows']}개 · "
                "반복 사용된 개발 자료 (최종 성능 아님)")
    return experiment_card(title, subtitle, arms, path)


def _errors(groups) -> list[dict]:
    wrong = groups[predict_labels(groups.decision) != groups.label]
    return [dict(group_id=r.group_id, label=r.label, decision=round(float(r.decision), 4))
            for r in wrong.itertuples()]


def _markdown(report: dict) -> str:
    lines = [
        "# MiniRocket + Ridge 개발 비교 (C)",
        "",
        f"- 패키지 manifest SHA-256: `{report['package_manifest_sha256']}`",
        f"- 입력 계약: `{report['contract_version']}` / `{report['contract_id']}`",
        f"- validation: 창 {report['validation']['windows']}개, 원본 영상 그룹 {report['validation']['groups']}개",
        "- validation 은 반복 사용된 개발 자료다. 최종 일반화 성능이 아니다.",
        "- 점수는 Ridge margin 이며 확률이 아니다.",
        "",
        "| 학습 구성 | 변환기 fit | alpha | 학습 창/그룹 | 창 macro-F1 | track macro-F1 | 그룹 macro-F1 (95% CI) | 새 recall | 드론 recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for arm, row in report["results"].items():
        v, ci = row["validation"], row["group_bootstrap"]["macro_f1_95ci"]
        lines.append(
            f"| {arm} | {row['representation_fit']} | {row['alpha']:g} | "
            f"{row['train_windows']}/{row['train_groups']} | {v['window']['macro_f1']:.4f} | "
            f"{v['track']['macro_f1']:.4f} | {v['group']['macro_f1']:.4f} ({ci[0]:.2f}–{ci[1]:.2f}) | "
            f"{v['group']['recall']['bird']:.4f} | {v['group']['recall']['drone']:.4f} |"
        )
    lines += ["", "## 그룹 단위 오분류", ""]
    for arm, row in report["results"].items():
        errors = ", ".join(f"{e['group_id']}({e['label']}, {e['decision']:+.3f})" for e in row["errors"]) or "없음"
        lines.append(f"- {arm}: {errors}")
    lines += ["", "## 추론 지연", ""]
    for arm, row in report["results"].items():
        lines.append(f"- {arm}: 첫 호출 {row['latency_ms']['first_call']} ms, "
                     f"준비 후 창당 {row['latency_ms']['per_window_warm']} ms")
    if report["margin_threshold"] > 0:
        lines += ["", f"## 보류 정책 (|margin| < {report['margin_threshold']:g})", ""]
        for arm, row in report["results"].items():
            a = row["validation"]["group"]["abstain"]
            lines.append(f"- {arm}: coverage {a['coverage']:.3f}, 판정 정확도 {a['decided_accuracy']}, "
                         f"보류 포함 recall {a['recall_with_abstain']}")
    return "\n".join(lines) + "\n"


def publish(report_dir: Path = _DEFAULT_REPORT_DIR) -> None:
    """검토한 실행 결과를 docs/ 확정본으로 복사한다. 지표 변화가 커밋 diff 로 드러난다."""
    shutil.copy2(report_dir / "metrics.json", _DOCS_DIR / "minirocket_metrics.json")
    shutil.copy2(report_dir / "REPORT.md", _DOCS_DIR / "minirocket_evaluation.md")
    (_DOCS_DIR / "images").mkdir(exist_ok=True)
    shutil.copy2(report_dir / "card.png", _DOCS_DIR / "images" / "minirocket_evaluation.png")
    print(f"확정본 승격: {_DOCS_DIR / 'minirocket_metrics.json'}, {_DOCS_DIR / 'minirocket_evaluation.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRocket + Ridge 세 학습 구성 비교")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--margin-threshold", type=float, default=0.0,
                        help="|margin| 이 이 값보다 작으면 uncertain 으로 보류 (0 이면 보류 없음)")
    parser.add_argument("--publish", action="store_true", help="결과를 docs/ 에 확정본으로 복사")
    args = parser.parse_args()
    if args.margin_threshold < 0 or not np.isfinite(args.margin_threshold):
        parser.error("--margin-threshold must be a finite value >= 0")
    evaluate(args.package, args.report_dir, args.margin_threshold)
    if args.publish:
        publish(args.report_dir)
