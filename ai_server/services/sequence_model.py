"""Part C: MiniRocket + StandardScaler + RidgeClassifier 공통 모듈.

train.py, evaluate.py, classifier.py 가 같은 함수를 쓰도록 모델 생성·데이터 로드·
점수 집계·지표 계산을 한 곳에 모은다. 근거 문서는
research/MINIROCKET_C_PROPOSAL.md, research/REAL_REFERENCE_DATASET_V1.md,
research/REAL_REFERENCE_COMPARISON_B3.md 이다.

지켜야 할 규칙 (B→C 계약):
- 입력은 trajectory-sequence-1.0.1 의 float32 (N, 4, 60) 배열만 쓴다.
  y, group_id, domain, 품질 메타데이터는 모델 입력 채널이 아니다.
- MiniRocket·scaler·Ridge 는 학습 자료에만 fit 한다. validation 에 fit 하지 않는다.
- Ridge 의 decision_function 은 확률이 아니다. 양수면 classes[1] = "drone".
- 분할·CV 는 원본 영상(합성은 비행 seed) 그룹 단위로 한다. 창 단위로 섞지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from research.trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig

LABELS = ["bird", "drone"]
N_CHANNELS = len(CHANNELS)
N_SAMPLES = SequenceConfig().samples
CONTRACT_ID = SequenceConfig().fingerprint

# B-3 비교와 같은 설정을 써야 C 결과가 B 결과를 재현하는지 확인할 수 있다.
N_KERNELS = 10_000
RANDOM_STATE = 20260928
ALPHAS = (0.1, 1.0, 10.0, 100.0)
CV_SPLITS = 3

ARMS = ("real_only", "real_plus_augmentation", "synthetic_only")
# 변환기(MiniRocket+scaler)를 어느 자료에 fit 하는지. 증강 arm 은 실제 train 에만
# fit 하고 증강 창은 Ridge 학습에만 넣는다. 합성-only 는 합성에만 fit 해야
# 진짜 합성-only 대조군이 된다.
REPRESENTATION_FIT = {
    "real_only": "real_only",
    "real_plus_augmentation": "real_only",
    "synthetic_only": "synthetic_only",
}
# alpha 를 어느 arm 의 train 그룹 CV 로 고르는지. 실제 두 arm 은 같은 alpha 를 공유한다.
ALPHA_SOURCE = {
    "real_only": "real_only",
    "real_plus_augmentation": "real_only",
    "synthetic_only": "synthetic_only",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = PROJECT_ROOT / "research" / "output" / "real_reference_comparison_v1"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "minirocket_classifier.joblib"
SCORE_TYPE = "ridge_margin_mean"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_windows(x: np.ndarray) -> np.ndarray:
    """계약 위반 입력이 모델까지 들어가지 않게 막는다."""
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (N_CHANNELS, N_SAMPLES):
        raise ValueError(f"Expected float32 (N,{N_CHANNELS},{N_SAMPLES}), got {x.dtype} {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in sequence windows")
    if len(x) and (
        not np.allclose(x[:, 2:, 0], 0)
        or not np.allclose(x[:, 2:, 1:], np.diff(x[:, :2], axis=2), atol=1e-6)
    ):
        raise ValueError("Displacement channels do not match coordinate channels")
    return x


def load_arrays(path: Path, *, with_weights: bool) -> dict[str, np.ndarray]:
    keys = ["X", "y", "group_id", "sample_id"]
    if with_weights:
        keys += ["sample_weight", "domain"]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in keys}
    validate_windows(data["X"])
    n = len(data["X"])
    if any(len(value) != n for value in data.values()):
        raise ValueError(f"Misaligned arrays: {path.name}")
    if set(data["y"]) != set(LABELS):
        raise ValueError(f"Both labels required: {path.name}")
    if len(set(data["sample_id"])) != n:
        raise ValueError(f"Duplicate sample IDs: {path.name}")
    if with_weights:
        weights = data["sample_weight"]
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"Invalid sample_weight: {path.name}")
    return data


def load_package(folder: Path = DEFAULT_PACKAGE):
    """B 가 넘긴 real_reference_comparison 패키지를 해시·계보까지 검사해 읽는다."""
    folder = Path(folder)
    manifest_path = folder / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"데이터 패키지가 없습니다: {folder}\n"
            "real_reference_comparison_v1.zip 을 research/output/ 아래에 풀어주세요."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["schema"] != "real-reference-comparison-1"
        or manifest["contract_version"] != CONTRACT_VERSION
        or manifest["contract_id"] != CONTRACT_ID
        or manifest["test_included"]
        or manifest["source_group_overlap"]
    ):
        raise ValueError("Unexpected dataset contract or split")
    for name, checksum in manifest["file_hashes"].items():
        if file_hash(folder / name) != checksum:
            raise ValueError(f"Package file changed: {name}")

    train = {}
    for arm in ARMS:
        name = manifest["arms"][arm]
        train[arm] = load_arrays(folder / name, with_weights=True)
        if manifest["counts"][name]["windows"] != len(train[arm]["y"]):
            raise ValueError(f"Training count changed: {arm}")
    validation_name = manifest["common_validation"]
    validation = load_arrays(folder / validation_name, with_weights=False)
    if manifest["counts"][validation_name]["windows"] != len(validation["y"]):
        raise ValueError("Validation count changed")

    real, augmented, synthetic = (train[arm] for arm in ARMS)
    n = len(real["y"])
    if (
        not np.array_equal(real["sample_id"], augmented["sample_id"][:n])
        or not np.array_equal(real["X"], augmented["X"][:n])
        or set(augmented["domain"][:n]) != {"real"}
        or set(augmented["domain"][n:]) != {"real_anchor_augmented"}
        or set(synthetic["domain"]) != {"synthetic"}
        or set(augmented["group_id"][n:]) - set(real["group_id"])
        or set(synthetic["group_id"]) & set(real["group_id"])
    ):
        raise ValueError("Training cohort lineage changed")
    for arm, data in train.items():
        if set(data["group_id"]) & set(validation["group_id"]):
            raise ValueError(f"Train/validation group overlap: {arm}")
        if set(data["sample_id"]) & set(validation["sample_id"]):
            raise ValueError(f"Train/validation sample overlap: {arm}")
    aug_mass = float(augmented["sample_weight"][n:].sum() / augmented["sample_weight"].sum())
    if not np.isclose(aug_mass, manifest["augmentation_sample_weight_mass"]):
        raise ValueError("Augmentation weight mass changed")
    return manifest, train, validation


def make_rocket():
    # aeon 은 numba 를 끌어오므로 import 비용이 크다. 실제로 쓸 때만 불러온다.
    from aeon.transformations.collection.convolution_based import MiniRocket

    return MiniRocket(n_kernels=N_KERNELS, random_state=RANDOM_STATE, n_jobs=1)


def fit_representation(x: np.ndarray):
    rocket = make_rocket()
    scaler = StandardScaler(with_mean=False)
    scaler.fit(rocket.fit_transform(validate_windows(x)))
    return rocket, scaler


def transform(rocket, scaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(rocket.transform(validate_windows(x))).astype(np.float64)


def track_ids(sample_ids) -> np.ndarray:
    """`track-xxx:w0003` → `track-xxx`. 증강 접미사(`:anchorNNN`)도 부모 track 으로 묶는다."""
    return np.array([str(s).split(":w", 1)[0] for s in sample_ids])


def aggregate_scores(data: dict, decisions: np.ndarray):
    """창 → track(창 점수 평균) → 원본 영상 그룹(track 점수 평균) 순으로 집계한다.

    긴 영상이 창 개수만큼 가산점을 받지 않도록 각 단계에서 평균을 쓴다.
    """
    windows = pd.DataFrame(
        dict(
            sample_id=data["sample_id"],
            track_id=track_ids(data["sample_id"]),
            group_id=data["group_id"],
            label=data["y"],
            decision=decisions,
        )
    )
    tracks = windows.groupby(["group_id", "track_id", "label"], as_index=False).decision.mean()
    if (tracks.groupby("group_id").label.nunique() > 1).any():
        raise ValueError("Mixed-label source group; group-level decision undefined")
    groups = tracks.groupby(["group_id", "label"], as_index=False).decision.mean()
    return windows, tracks, groups


def predict_labels(decisions, margin_threshold: float = 0.0) -> np.ndarray:
    decisions = np.asarray(decisions, dtype=float)
    labels = np.where(decisions >= 0, "drone", "bird").astype(object)
    labels[np.abs(decisions) < margin_threshold] = "uncertain"
    return labels


def metrics(table: pd.DataFrame, margin_threshold: float = 0.0) -> dict:
    """보류 없는 분류 지표 + 보류 정책 적용 시 coverage / 판정 정확도."""
    truth = table.label.to_numpy()
    decisions = table.decision.to_numpy()
    predicted = predict_labels(decisions)
    result = dict(
        n=len(table),
        accuracy=float(accuracy_score(truth, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
        recall=dict(zip(LABELS, recall_score(
            truth, predicted, labels=LABELS, average=None, zero_division=0).tolist())),
        roc_auc=float(roc_auc_score(truth == "drone", decisions)) if len(set(truth)) == 2 else None,
        confusion_matrix=confusion_matrix(truth, predicted, labels=LABELS).tolist(),
    )
    if margin_threshold > 0:
        gated = predict_labels(decisions, margin_threshold)
        decided = gated != "uncertain"
        result["abstain"] = dict(
            margin_threshold=margin_threshold,
            coverage=float(decided.mean()),
            decided_accuracy=float((gated[decided] == truth[decided]).mean()) if decided.any() else None,
            # 보류를 오답으로 세는 클래스별 recall. 보류를 분모에서 숨기지 않는다.
            recall_with_abstain={
                label: float((gated[truth == label] == label).mean()) for label in LABELS
            },
        )
    return result


def group_bootstrap_ci(groups: pd.DataFrame, draws: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """원본 영상 그룹을 재표집한 macro-F1 / balanced accuracy 95% 구간.

    창을 독립 표본으로 재표집하면 구간이 과하게 좁아지므로 그룹 단위로만 뽑는다.
    """
    rng = np.random.default_rng(seed)
    truth = groups.label.to_numpy()
    predicted = predict_labels(groups.decision.to_numpy())
    f1s, bals = [], []
    for _ in range(draws):
        index = rng.integers(0, len(groups), len(groups))
        t, p = truth[index], predicted[index]
        if len(set(t)) < 2:
            continue
        f1s.append(f1_score(t, p, labels=LABELS, average="macro", zero_division=0))
        bals.append(balanced_accuracy_score(t, p))
    interval = lambda values: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return dict(draws=len(f1s), macro_f1_95ci=interval(f1s), balanced_accuracy_95ci=interval(bals))


def balanced_group_weights(y, groups) -> np.ndarray:
    """클래스 총량을 같게, 클래스 안에서는 원본 영상 그룹마다 같은 총량을 준다.

    긴 드론 영상이 창 개수만큼 CV 학습을 지배하지 않게 한다. 총합은 창 수와 같다.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    weights = np.zeros(len(y))
    for label in LABELS:
        unique = np.unique(groups[y == label])
        for group in unique:
            mask = (y == label) & (groups == group)
            weights[mask] = len(y) / len(LABELS) / len(unique) / mask.sum()
    return weights


def choose_alpha(train: dict, alphas=ALPHAS):
    """train 그룹 CV 로 alpha 를 고른다. fold 마다 변환기를 새로 fit 해 누수를 막는다."""
    splitter = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (fit, held) in enumerate(splitter.split(train["X"], train["y"], train["group_id"])):
        if set(train["group_id"][fit]) & set(train["group_id"][held]):
            raise ValueError("Group leak inside CV")
        if set(train["y"][fit]) != set(LABELS) or set(train["y"][held]) != set(LABELS):
            raise ValueError("A CV fold is missing a class")
        rocket, scaler = fit_representation(train["X"][fit])
        z_fit = transform(rocket, scaler, train["X"][fit])
        z_held = transform(rocket, scaler, train["X"][held])
        weight = balanced_group_weights(train["y"][fit], train["group_id"][fit])
        part = {key: value[held] for key, value in train.items()}
        for alpha in alphas:
            ridge = RidgeClassifier(alpha=alpha).fit(z_fit, train["y"][fit], sample_weight=weight)
            groups = aggregate_scores(part, ridge.decision_function(z_held))[2]
            rows.append(dict(fold=fold, alpha=alpha, group_macro_f1=metrics(groups)["macro_f1"]))
    table = pd.DataFrame(rows)
    # 동률이면 가장 작은 alpha 를 고른다 (B-3 evaluate_sequence_handoff 와 같은 규칙).
    best = float(table.groupby("alpha").group_macro_f1.mean().idxmax())
    return best, table


def fit_arm(train: dict, arm: str, alpha: float) -> dict:
    """한 학습 구성의 변환기+scaler+Ridge 를 묶은 모델 번들을 만든다."""
    data = train[arm]
    rocket, scaler = fit_representation(train[REPRESENTATION_FIT[arm]]["X"])
    classifier = RidgeClassifier(alpha=alpha).fit(
        transform(rocket, scaler, data["X"]), data["y"], sample_weight=data["sample_weight"])
    if classifier.classes_.tolist() != LABELS:
        raise ValueError("Unexpected class order")
    return dict(
        rocket=rocket,
        scaler=scaler,
        classifier=classifier,
        classes=LABELS,
        alpha=alpha,
        arm=arm,
        representation_fit=REPRESENTATION_FIT[arm],
        contract_version=CONTRACT_VERSION,
        contract_id=CONTRACT_ID,
        channels=list(CHANNELS),
        score_type=SCORE_TYPE,
        score_is_probability=False,
        train_windows=len(data["y"]),
        train_groups=len(set(data["group_id"])),
        versions=library_versions(),
    )


def decision_scores(model: dict, x: np.ndarray) -> np.ndarray:
    check_bundle(model)
    if not len(x):
        return np.empty(0)
    return model["classifier"].decision_function(transform(model["rocket"], model["scaler"], x))


def check_bundle(model: dict) -> None:
    if model.get("contract_version") != CONTRACT_VERSION or model.get("contract_id") != CONTRACT_ID:
        raise ValueError("Model/preprocessing contract mismatch")
    if list(model.get("classes", [])) != LABELS:
        raise ValueError("Unexpected class order; score sign would be wrong")


def library_versions() -> dict[str, str]:
    names = ("aeon", "numba", "numpy", "scikit-learn", "scipy", "joblib")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions
