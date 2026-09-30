"""Build real-only MiniRocket inputs with auditable provisional source groups.

python -m research.build_real_sequence_dataset --output research/output/real_sequences_v1
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from .io import write_json
from .trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig, window_track
from .verify_real_track import load_tracks


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_inputs(input_dir, audit_path=None):
    audit = {}
    if audit_path and Path(audit_path).exists():
        for row in json.loads(Path(audit_path).read_text(encoding="utf-8-sig")):
            audit[(row["file_sha256"], row["history_hash"])] = row
    records, errors = [], []
    for label in ("bird", "drone"):
        for path in sorted((Path(input_dir) / label).glob("*.json")):
            checksum = file_hash(path)
            try:
                tracks = load_tracks(path)
                for index, track in enumerate(tracks):
                    history_hash = digest(track["history"])
                    old = audit.get((checksum, history_hash), {})
                    observations = [[p.get(k) for k in ("frame_index", "timestamp_ms", "cx", "cy")]
                                    for p in track["history"]]
                    center_hash = digest(dict(points=observations, width=track.get("processed_width"),
                                              height=track.get("processed_height")))
                    track_id = "track-" + digest([label, path.name, index, checksum])[:16]
                    records.append(dict(parent_track_id=track_id, label=label, file_name=path.name,
                                        source_path=str(path.resolve()), file_sha256=checksum,
                                        source_video_id=track.get("source_video_id", ""),
                                        history_hash=history_hash, center_hash=center_hash,
                                        video_sha256=old.get("video_sha256", ""),
                                        audit_sample_id=old.get("sample_id", ""),
                                        audit_hash_match=bool(old), track=track,
                                        processed_width=track.get("processed_width"),
                                        processed_height=track.get("processed_height"),
                                        stabilization_applied=track.get("stabilization", {}).get("applied"),
                                        group_review="provisional_session_unverified"))
            except (ValueError, KeyError, TypeError) as error:
                errors.append(dict(source_path=str(path.resolve()), file_sha256=checksum,
                                   reason="invalid_input", detail=str(error)))
    if not records:
        raise ValueError("No readable real tracks found")
    return records, errors


def group_records(records, overrides=None):
    """Union evidence across labels; filename hints remain explicitly provisional."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    overrides = overrides or {}
    unknown = set(overrides) - {r["parent_track_id"] for r in records}
    if unknown:
        raise ValueError(f"Unknown track IDs in group overrides: {sorted(unknown)}")
    seen, links = {}, []
    for i, rec in enumerate(records):
        family = re.sub(r"_track_?sequence$", "", Path(rec["file_name"]).stem, flags=re.I)
        family = re.sub(r"\(\d+\)$", "", family)
        family = re.sub(r"_(?:위에|아래)[ _]?새$", "", family)
        keys = [("source_id", rec["source_video_id"]), ("centers", rec["center_hash"]),
                ("video_bytes", rec["video_sha256"]), ("filename_hint", rec["label"] + ":" + family)]
        override = overrides.get(rec["parent_track_id"], "")
        if override:
            keys.append(("reviewed_session", override))
            rec["group_review"] = "session_override_supplied"
        for kind, value in keys:
            if not value:
                continue
            key = (kind, value)
            if key in seen:
                other = seen[key]
                parent[find(i)] = find(other)
                links.append(dict(left=rec["parent_track_id"], right=records[other]["parent_track_id"],
                                  evidence=kind, value=value))
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[find(i)].append(rec)
    for members in groups.values():
        gid = "group-" + digest(sorted(r["parent_track_id"] for r in members))[:16]
        for rec in members:
            rec["source_group_id"] = gid
            rec["group_size"] = len(members)
    return links


def assign_splits(records, seed):
    groups = defaultdict(set)
    for rec in records:
        groups[rec["source_group_id"]].add(rec["label"])
    strata = defaultdict(list)
    for group, labels in groups.items():
        strata["+".join(sorted(labels))].append(group)
    assignments = {}
    for label, ids in sorted(strata.items()):
        # Mixed-label source groups remain intact; small strata stay in train.
        ids = sorted(ids, key=lambda gid: digest([seed, label, gid]))
        n_eval = max(1, int(round(len(ids) * 0.2))) if len(ids) >= 5 else 0
        for i, gid in enumerate(ids):
            assignments[gid] = ("validation" if i < n_eval else
                                "test" if i < 2 * n_eval else "train")
    for rec in records:
        rec["split"] = assignments[rec["source_group_id"]]


def _write_table(path, rows, columns):
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def build(input_dir, output_dir, audit_path=None, overrides_path=None, seed=20260927,
          config=None, max_train_windows_per_group=8):
    config = config or SequenceConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing datasets are immutable")
    if max_train_windows_per_group < 1:
        raise ValueError("Training group cap must be positive")
    records, errors = read_inputs(input_dir, audit_path)
    overrides = {}
    if overrides_path:
        table = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
        if table.parent_track_id.duplicated().any():
            raise ValueError("Duplicate override track IDs")
        overrides = dict(zip(table.parent_track_id, table.session_group))
    links = group_records(records, overrides)
    assign_splits(records, seed)
    windows, excluded = [], list(errors)
    canonical = {}
    # Exact coordinate duplicates are not extra observations, even within a split.
    for rec in sorted(records, key=lambda r: r["parent_track_id"]):
        identity = rec["center_hash"]
        if identity in canonical:
            if rec["label"] != canonical[identity]["label"]:
                raise ValueError("Identical trajectory has conflicting class labels")
            rec.update(status="duplicate", duplicate_of=canonical[identity]["parent_track_id"],
                       accepted_windows=0, exported_windows=0)
            continue
        canonical[identity] = rec
        try:
            accepted, rejected = window_track(rec["track"], config)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            accepted, rejected = [], [dict(reason="invalid_track", detail=str(error))]
        rec.update(status="accepted" if accepted else "excluded", accepted_windows=len(accepted))
        base = {k: rec[k] for k in ("parent_track_id", "source_group_id", "split", "label",
                                    "file_name", "audit_sample_id")}
        for row in accepted:
            row.update(base, domain="real", sample_id=rec["parent_track_id"] + f":w{row['window_index']:04d}")
            windows.append(row)
        excluded.extend(dict(base, **row) for row in rejected)

    selected = []
    by_group = defaultdict(list)
    for row in windows:
        by_group[row["source_group_id"]].append(row)
    for gid, rows in sorted(by_group.items()):
        rows.sort(key=lambda row: (row["parent_track_id"], row["window_index"]))
        if rows[0]["split"] == "train" and len(rows) > max_train_windows_per_group:
            indices = set(np.linspace(0, len(rows) - 1, max_train_windows_per_group, dtype=int).tolist())
            selected.extend(row for i, row in enumerate(rows) if i in indices)
            excluded.extend({**{k: v for k, v in row.items() if k != "X"}, "reason": "training_group_cap"}
                            for i, row in enumerate(rows) if i not in indices)
        else:
            selected.extend(rows)
    selected.sort(key=lambda row: (row["split"], row["sample_id"]))
    counts = Counter(row["parent_track_id"] for row in selected)
    for rec in records:
        rec["exported_windows"] = counts[rec["parent_track_id"]]
    for rec in records:
        if file_hash(rec["source_path"]) != rec["file_sha256"]:
            raise ValueError("Input changed during build")
    output.mkdir(parents=True, exist_ok=True)
    metadata, summary = [], []
    for split in ("train", "validation", "test"):
        rows = [r for r in selected if r["split"] == split]
        x = np.stack([r["X"] for r in rows]) if rows else np.empty((0, 4, config.samples), np.float32)
        arrays = dict(X=x, y=np.asarray([r["label"] for r in rows], dtype="U5"),
                      sample_id=np.asarray([r["sample_id"] for r in rows], dtype="U40"),
                      group_id=np.asarray([r["source_group_id"] for r in rows], dtype="U32"))
        if not np.isfinite(x).all():
            raise ValueError("Non-finite model input")
        np.savez_compressed(output / f"{split}.npz", **arrays)
        for index, row in enumerate(rows):
            metadata.append(dict(npz_row=index, **{k: v for k, v in row.items() if k != "X"}))
        for label in ("bird", "drone"):
            subset = [r for r in rows if r["label"] == label]
            summary.append(dict(split=split, label=label, windows=len(subset),
                                tracks=len({r["parent_track_id"] for r in subset}),
                                groups=len({r["source_group_id"] for r in subset})))
    groups_per_split = {s: {r["source_group_id"] for r in selected if r["split"] == s}
                        for s in ("train", "validation", "test")}
    overlap = sum(len(groups_per_split[a] & groups_per_split[b])
                  for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    if overlap:
        raise ValueError("Group leakage")
    inventory = [{k: v for k, v in rec.items() if k != "track"} for rec in records]
    _write_table(output / "metadata.csv", metadata, ["sample_id", "npz_row", "split"])
    _write_table(output / "source_inventory.csv", inventory, ["parent_track_id"])
    _write_table(output / "group_evidence.csv", links, ["left", "right", "evidence", "value"])
    _write_table(output / "excluded_windows.csv", excluded, ["parent_track_id", "reason"])
    _write_table(output / "split_summary.csv", summary, ["split", "label", "windows"])
    _write_table(output / "session_review.csv", [dict(parent_track_id=r["parent_track_id"],
                 file_name=r["file_name"], source_group_id=r["source_group_id"], session_group="")
                 for r in inventory], ["parent_track_id", "session_group"])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = dict(contract_version=CONTRACT_VERSION, contract_id=config.fingerprint,
                    created_at=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                    config=asdict(config), channels=list(CHANNELS), shape=["N", 4, config.samples],
                    dtype="float32", labels=["bird", "drone"], seed=seed,
                    coordinate_space="A exported stabilized centers, clipped by A",
                    split_policy="60/20/20 approximate by provisional source groups per class",
                    evaluation_status="development_only_previously_inspected_not_untouched_holdout",
                    group_status="provisional; original acquisition/session identity not fully verified",
                    independent_evaluation_ready=False,
                    scope="folder labels; drone subtype and behavior not manually verified",
                    scale_floor_policy="fixed engineering default, not fitted or empirically calibrated",
                    augmentation=False, simulator_used=False,
                    max_train_windows_per_group=max_train_windows_per_group,
                    source_files=len({r["source_path"] for r in inventory}), tracks=len(inventory),
                    source_groups=len({r["source_group_id"] for r in inventory}),
                    split_group_overlap=overlap, input_errors=errors, summary=summary,
                    all_splits_have_both_classes=all(r["windows"] > 0 for r in summary),
                    environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                    excluded_reasons=dict(Counter(r["reason"] for r in excluded)),
                    sources=inventory,
                    code_hashes={p.name: file_hash(p) for p in
                                 (Path(__file__), Path(__file__).with_name("trajectory_sequence.py"))})
    write_report(output, manifest, metadata)
    plot_examples(output, selected, config)
    manifest["artifact_hashes"] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def plot_examples(output, windows, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples, seen = [], set()
    for label in ("bird", "drone"):
        count = 0
        for row in windows:
            if row["label"] == label and row["source_group_id"] not in seen:
                examples.append(row)
                seen.add(row["source_group_id"])
                count += 1
                if count == 2:
                    break
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3 * len(examples)), squeeze=False,
                             layout="constrained")
    for (path_ax, time_ax), row in zip(axes, examples):
        x = row["X"]
        path_ax.plot(x[0], x[1], color="#167a95")
        path_ax.scatter(x[0, 0], x[1, 0], color="#ce5736", label="start")
        path_ax.set_aspect("equal", adjustable="datalim")
        path_ax.invert_yaxis()
        name = row["audit_sample_id"] or row["parent_track_id"][:12]
        path_ax.set(title=f"{row['label']} / {name} / {row['split']} / t={row['relative_start_s']:.0f}s",
                    xlabel="q_x", ylabel="q_y (screen down)")
        path_ax.legend(fontsize=8)
        for channel, color in zip(range(4), ("#167a95", "#ce5736", "#438743", "#9b4a91")):
            time_ax.plot(np.arange(config.samples)/config.fps, x[channel], label=CHANNELS[channel], color=color)
        time_ax.set(xlabel="time within window (s)", ylabel="model input", title="Four channels, no additional SG filter")
        time_ax.legend(ncol=4, fontsize=8)
        path_ax.grid(alpha=.2)
        time_ax.grid(alpha=.2)
    fig.savefig(output / "sequence_examples.png", dpi=140)
    plt.close(fig)


def write_report(output, manifest, metadata):
    lines = ["# 실제 A 궤적 시계열 데이터셋", "", "## 생성 결과", "",
             f"- 계약: `{manifest['contract_version']}` / `{manifest['contract_id']}`",
             "- 2초, 30Hz, 60샘플, 채널 q_x/q_y/d_x/d_y. 실제 자료만 사용했다.",
             f"- 입력 {manifest['source_files']}개 파일, 잠정 원본 그룹 {manifest['source_groups']}개.",
             f"- 확인된 그룹의 split 교차: {manifest['split_group_overlap']}건.",
             "- 촬영 세션 독립성은 미확인이다. 같은 촬영의 재인코딩/별도 클립을 자동으로 모두 찾을 수 없다.",
             "- train/validation/test 모두 기존에 살펴본 개발 자료에서 분리했다. 미사용 최종 test가 아니다.",
             "- 정답은 폴더 기준이며 추적 대상·기체 종류·행동의 수동 검증을 대신하지 않는다.", "",
             "| split | label | 원본 그룹 | track | 창 |", "|---|---|---:|---:|---:|"]
    for row in manifest["summary"]:
        lines.append(f"| {row['split']} | {row['label']} | {row['groups']} | {row['tracks']} | {row['windows']} |")
    lines += ["", "## 제외 및 품질", "", "| 사유 | 건수 |", "|---|---:|"]
    for reason, count in manifest["excluded_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    flags = Counter(flag for row in metadata for flag in row["quality_flags"].split(";") if flag)
    for flag, count in flags.items():
        lines.append(f"| 유지한 창의 표시: {flag} | {count} |")
    lines += ["", "## 사용 방법", "", "```python", "import numpy as np",
              "data = np.load('train.npz', allow_pickle=False)",
              "X, y = data['X'], data['y']", "groups = data['group_id']", "```", "",
              "- NPZ의 sample_id와 metadata.csv의 sample_id를 조인한다. npz_row는 split별 배열 행 번호다.",
              "- source_inventory.csv는 제외·중복 원본까지 포함한다. group_evidence.csv는 묶은 근거다.",
              "- session_review.csv에 실제 같은 촬영의 session_group을 입력하고 새 폴더에 재생성할 수 있다.",
              "- 자동 근거는 수동 그룹 입력으로 쪼개지지 않는다. 누수 위험을 줄이기 위해 추가 병합만 한다.",
              "- train 원본 그룹당 최대 8개 창을 고르게 선택했다. 평가 창은 모두 보존했다.",
              "- boundary_contact와 low_spatial_extent는 유지한 품질 표시이며 물리적 정지나 A 오류의 확정이 아니다.",
              "- scale_floor=0.0025는 영상 너비 단위의 초기 공학적 설정이며 실제 잡음에서 추정한 값이 아니다.",
              "- 소규모 실제 자료이므로 6000/2000/2000개로 복제하지 않았다. 합성·증강·모델 학습은 다음 단계다.",
              "- C는 MiniRocket/scaler를 train에만 fit한다. 같은 그룹의 창을 다시 무작위 분할하지 않는다.",
              "- 추론에 필요한 2초 관측이 쌓이기 전에는 관측 부족 상태로 처리한다."]
    if metadata:
        lines += ["", "## 변환 예시", "", "![시계열 입력 예시](sequence_examples.png)", "",
                  "각 클래스에서 서로 다른 원본 그룹의 첫 두 사례를 표시한다. 대표성이나 분류 가능성을 입증하는 그림은 아니다."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("research/data/tracks"))
    parser.add_argument("--output", type=Path, default=Path("research/output/real_sequences_v1"))
    parser.add_argument("--audit", type=Path, default=Path("research/output/a_preprocessing_audit/manifest.json"))
    parser.add_argument("--session-overrides", type=Path)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    manifest = build(args.input, args.output, args.audit, args.session_overrides, args.seed)
    print(json.dumps(dict(output=str(args.output), summary=manifest["summary"],
                          exclusions=manifest["excluded_reasons"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
