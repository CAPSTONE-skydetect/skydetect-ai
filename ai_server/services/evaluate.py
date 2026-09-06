"""Part C: RF 분류기 성능 평가 스크립트.

주 지표는 train 학습 → test 홀드아웃 평가이며,
보조 지표로 train 내부 GroupKFold(family_id) 교차검증을 함께 산출한다.

피처 목록은 train.FEATURE_NAMES를 참조하므로, 피처가 바뀌어도
이 스크립트를 수정할 필요가 없다.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_score

from ai_server.services.train import (
    FEATURE_NAMES,
    MIN_SAMPLES_LEAF,
    N_ESTIMATORS,
    RANDOM_STATE,
    build_classifier,
    extract_provenance,
    load_dataset,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAIN_PATH = str(_PROJECT_ROOT / "data" / "train_features.csv")
_DEFAULT_TEST_PATH = str(_PROJECT_ROOT / "data" / "test_features.csv")
_DEFAULT_REPORT_DIR = str(_PROJECT_ROOT / "reports")

# 양성 클래스. ROC-AUC / PR-AUC 계산 기준이다.
_POSITIVE_LABEL = "drone"

# 정확도를 분해해서 볼 세그먼트 컬럼. 없는 컬럼은 자동으로 건너뛴다.
_SEGMENT_COLUMNS = [
    "observation_profile",
    "scenario",
    "behavior_mode",
    "training_length_group",
]

# 교차검증 시 누수를 막기 위한 그룹 컬럼.
_GROUP_COLUMN = "family_id"


def _check_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """train/test 사이의 누수 여부를 검사한다."""
    result: dict[str, Any] = {}
    for column in [_GROUP_COLUMN, "sample_id"]:
        if column in train_df.columns and column in test_df.columns:
            overlap = set(train_df[column]) & set(test_df[column])
            result[f"{column}_overlap"] = len(overlap)
    result["duplicate_feature_rows"] = int(
        pd.merge(train_df[FEATURE_NAMES], test_df[FEATURE_NAMES]).shape[0]
    )
    return result


def _segment_accuracy(
    test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """세그먼트별 정확도를 분해한다. 취약 시나리오를 찾기 위함이다."""
    segments: dict[str, Any] = {}
    for column in _SEGMENT_COLUMNS:
        if column not in test_df.columns:
            continue
        breakdown = {}
        for value in sorted(test_df[column].dropna().unique()):
            mask = (test_df[column] == value).to_numpy()
            breakdown[str(value)] = {
                "n": int(mask.sum()),
                "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
            }
        segments[column] = breakdown
    return segments


def _plot(
    report_dir: Path,
    cm: np.ndarray,
    classes: list[str],
    accuracy: float,
    y_binary: np.ndarray,
    proba: np.ndarray,
    roc_auc: float,
    importance: list[tuple[str, float]],
) -> None:
    """confusion matrix / ROC / feature importance를 한 장에 그린다."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(len(classes)), classes)
    axes[0].set_yticks(range(len(classes)), classes)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix (acc={accuracy:.4f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )

    fpr, tpr, _ = roc_curve(y_binary, proba)
    axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    names = [name for name, _ in importance][::-1]
    values = [value for _, value in importance][::-1]
    axes[2].barh(names, values)
    axes[2].set_title("Feature Importance")
    axes[2].tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(report_dir / "report.png", dpi=130)
    plt.close(fig)


def evaluate(
    train_path: str = _DEFAULT_TRAIN_PATH,
    test_path: str = _DEFAULT_TEST_PATH,
    report_dir: str = _DEFAULT_REPORT_DIR,
    n_estimators: int = N_ESTIMATORS,
) -> dict[str, Any]:
    """train으로 학습하고 test 홀드아웃으로 평가해 지표를 산출한다."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, train_df = load_dataset(train_path)
    X_test, y_test, test_df = load_dataset(test_path)

    clf = build_classifier(n_estimators)
    clf.fit(X_train, y_train)

    classes = list(clf.classes_)
    positive_index = classes.index(_POSITIVE_LABEL)
    proba = clf.predict_proba(X_test)[:, positive_index]
    y_pred = clf.predict(X_test)
    y_binary = (y_test == _POSITIVE_LABEL).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    per_class = classification_report(y_test, y_pred, output_dict=True, digits=4)
    roc_auc = roc_auc_score(y_binary, proba)
    pr_auc = average_precision_score(y_binary, proba)

    # train 내부 교차검증. family 단위로 나눠 같은 궤적 계열이 양쪽에 걸치지 않게 한다.
    cv_scores = cross_val_score(
        build_classifier(n_estimators),
        X_train,
        y_train,
        groups=train_df[_GROUP_COLUMN],
        cv=GroupKFold(n_splits=5),
        scoring="f1_macro",
    )

    importance = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda pair: -pair[1]
    )

    report: dict[str, Any] = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_families": int(train_df[_GROUP_COLUMN].nunique()),
            "test_families": int(test_df[_GROUP_COLUMN].nunique()),
            "leakage_check": _check_leakage(train_df, test_df),
            "provenance": extract_provenance(train_df),
        },
        "model": {
            "n_estimators": n_estimators,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        },
        "holdout": {
            "accuracy": round(float(accuracy), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "per_class": {
                key: {metric: round(float(v), 4) for metric, v in value.items()}
                for key, value in per_class.items()
                if isinstance(value, dict)
            },
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        },
        "cv_train_groupkfold5_f1_macro": {
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
            "folds": [round(float(score), 4) for score in cv_scores],
        },
        "feature_importance": {name: round(float(v), 4) for name, v in importance},
        "segments": _segment_accuracy(test_df, y_test, y_pred),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(out_dir, cm, classes, accuracy, y_binary, proba, roc_auc, importance)

    return report


def _print_summary(report: dict[str, Any]) -> None:
    holdout = report["holdout"]
    cv = report["cv_train_groupkfold5_f1_macro"]

    print(f"\n{'=' * 52}")
    print("홀드아웃 테스트셋 평가")
    print(f"{'=' * 52}")
    print(f"  accuracy : {holdout['accuracy']:.4f}")
    print(f"  F1 (macro): {holdout['per_class']['macro avg']['f1-score']:.4f}")
    print(f"  ROC-AUC  : {holdout['roc_auc']:.4f}")
    print(f"  PR-AUC   : {holdout['pr_auc']:.4f}")
    print(f"\n  GroupKFold(5) F1 macro: {cv['mean']:.4f} ± {cv['std']:.4f}")

    print("\n  Confusion Matrix")
    labels = holdout["confusion_matrix"]["labels"]
    print(f"    {'':>10}" + "".join(f"{f'pred {c}':>14}" for c in labels))
    for label, row in zip(labels, holdout["confusion_matrix"]["matrix"]):
        print(f"    {f'true {label}':>10}" + "".join(f"{v:>14}" for v in row))

    print("\n  Feature Importance")
    for name, value in report["feature_importance"].items():
        bar = "█" * int(value * 100)
        print(f"    {name:<22} {value:.4f}  {bar}")

    for column, breakdown in report["segments"].items():
        print(f"\n  Accuracy by {column}")
        for value, stats in breakdown.items():
            print(f"    {value:<22} n={stats['n']:<6} acc={stats['accuracy']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF 분류기 성능 평가")
    parser.add_argument("--train-path", default=_DEFAULT_TRAIN_PATH, help="학습 CSV 경로")
    parser.add_argument("--test-path", default=_DEFAULT_TEST_PATH, help="테스트 CSV 경로")
    parser.add_argument(
        "--report-dir", default=_DEFAULT_REPORT_DIR, help="리포트 저장 디렉토리"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=N_ESTIMATORS, help="트리 개수"
    )
    args = parser.parse_args()

    result = evaluate(
        train_path=args.train_path,
        test_path=args.test_path,
        report_dir=args.report_dir,
        n_estimators=args.n_estimators,
    )
    _print_summary(result)
    print(f"\n리포트 저장: {args.report_dir}")
