"""Build a portable C dataset/model bundle and an honest development report."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .evaluate_sequence_handoff import balanced_group_weights, load_arrays, read_json
from .io import write_json
from .sequence_comparison import group_weights, quantile, sequence_metrics
from .trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig


def figures(folder, evaluation, target):
    cv = pd.read_csv(folder/"camera_cv.csv")
    validation = pd.read_csv(folder/"camera_validation.csv")
    results = read_json(evaluation/"results.json")
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    names = ["legacy","ground_span2.0","ground_span3.0"]
    for name,color in zip(names,["#737373","#147e92","#b26a29"]):
        part = cv[cv.variant == name]
        axes[0].plot(part.fold,part.distance,"o-",label=name,color=color)
    axes[0].set(xticks=[0,1,2],xlabel="Train group fold",ylabel="Normalized distribution distance",
                title="Camera correction did not improve overall CV")
    axes[0].legend(fontsize=8)
    x = np.arange(2)
    for offset,arm,color in [(-.18,"real_only","#147e92"),(.18,"real_plus_synthetic","#bf5a46")]:
        values = [results["arms"][arm][split]["group"]["macro_f1"] for split in ("validation","test")]
        axes[1].bar(x+offset,values,.35,label=arm,color=color)
        for position,value in zip(x+offset,values):
            axes[1].text(position,value+.02,f"{value:.3f}",ha="center",fontsize=9)
    axes[1].set(xticks=x,xticklabels=["Validation (10 videos)","Development test (10)"],ylim=(0,1.1),
                ylabel="Source-video macro-F1",title="Current synthetic addition reduced performance")
    axes[1].legend(fontsize=8,loc="lower left")
    span = validation[validation.metric == "screen_span"]
    for offset,variant,color in [(-.24,"real","#494949"),(0.,"legacy","#909090"),(.24,"ground_span2.0","#147e92")]:
        part = span[span.variant == ("legacy" if variant == "real" else variant)].set_index("label")
        values = part.loc[["bird","drone"],"real_median" if variant == "real" else "sim_median"].to_numpy()
        axes[2].bar(x+offset,values,.23,label=variant,color=color)
    axes[2].set(xticks=x,xticklabels=["bird","drone"],ylabel="2-second screen span / frame width",
                title="Group-weighted validation medians")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(target/"evaluation_summary.png",dpi=170)
    plt.close(fig)

    # Median-span examples, not hand-picked distinctive class patterns.
    real = load_arrays(folder/"real/train.npz")
    sim = load_arrays(folder/"synthetic_train.npz")
    real_meta = pd.read_csv(folder/"real/metadata.csv")
    real_meta = real_meta[real_meta.split == "train"].sort_values("npz_row")
    sim_meta = pd.read_csv(folder/"synthetic_train_metrics.csv")
    fig, axes = plt.subplots(2,4,figsize=(14,7),layout="constrained")
    for row,label in enumerate(("bird","drone")):
        for col,(data,meta,name) in enumerate(((real,real_meta,"Real train"),(sim,sim_meta,"Ground-camera synthetic"))):
            ids = np.flatnonzero(data["y"] == label)
            spans = np.array([sequence_metrics(data["X"][i],float(meta.iloc[i].normalization_scale))["screen_span"] for i in ids])
            index = ids[np.argsort(spans)[len(ids)//2]]
            q = data["X"][index,:2]
            raw = q * float(meta.iloc[index].normalization_scale)
            for ax,xy,limits,suffix in ((axes[row,col*2],raw,(-.5,.5),"screen scale"),
                                       (axes[row,col*2+1],q,(-2,2),"normalized shape")):
                ax.plot(xy[0],xy[1],lw=1,color="#147e92" if col == 0 else "#bf5a46")
                ax.scatter(xy[0,0],xy[1,0],s=20,color="black")
                ax.set(xlim=limits,ylim=limits[::-1],aspect="equal",title=f"{label}: {name}\n{suffix}")
                ax.grid(alpha=.2)
    fig.suptitle("Two-second median-span examples; screen plots centered for comparison, not original frame position")
    fig.savefig(target/"trajectory_examples.png",dpi=160)
    plt.close(fig)


def package(folder, evaluation_name="model_evaluation_float64"):
    folder = Path(folder)
    evaluation = folder/evaluation_name
    target = folder/"C_handoff"
    if target.exists() or (folder/"C_handoff.zip").exists():
        raise ValueError("Use a new handoff destination")
    result = read_json(evaluation/"results.json")
    preparation = read_json(folder/"preparation_results.json")
    audit = read_json(folder/"group_audit.json")
    source = read_json(folder/"real/dataset_manifest.json")
    freeze = read_json(evaluation/"model_selection.json")
    if file_hash(evaluation/(result["selected_arm"]+".joblib")) != result["selected_model_sha256"]:
        raise ValueError("Selected model checksum changed")
    target.mkdir()
    train, synthetic = load_arrays(folder/"real/train.npz"), load_arrays(folder/"synthetic_train.npz")
    n = len(train["y"])
    real_weights = balanced_group_weights(train["y"],train["group_id"],n)
    np.savez_compressed(target/"train_real.npz",**train,sample_weight=real_weights,domain=np.full(n,"real"))
    combined = {k:np.concatenate([train[k],synthetic[k]]) for k in train}
    mixed_weights = np.r_[real_weights*.75,balanced_group_weights(synthetic["y"],synthetic["group_id"],n*.25)]
    np.savez_compressed(target/"train_real_plus_synthetic.npz",**combined,sample_weight=mixed_weights,
                        domain=np.r_[np.full(n,"real"),np.full(len(synthetic["y"]),"synthetic")])
    for src,dst in ((folder/"synthetic_train.npz","synthetic_train_experimental.npz"),
                    (folder/"real/validation.npz","validation.npz"),(evaluation/"test.npz","test.npz"),
                    (evaluation/(result["selected_arm"]+".joblib"),"reference_model.joblib"),
                    (folder/"selected_profile.json","simulator_profile.json"),
                    (evaluation/"results.json","evaluation_results.json"),
                    (evaluation/"model_selection.json","model_selection.json"),
                    (evaluation/"protocol.json","evaluation_protocol.json")):
        shutil.copy2(src,target/dst)
    # No local source paths, named subjects or raw video files in the handoff.
    real_meta = pd.read_csv(folder/"real/metadata.csv")
    real_meta = real_meta.drop(columns=["file_name","audit_sample_id"],errors="ignore")
    real_meta.to_csv(target/"real_train_validation_metadata.csv",index=False)
    shutil.copy2(evaluation/"test.metadata.csv",target/"real_test_metadata.csv")
    sim_meta = pd.read_csv(folder/"synthetic_train_metrics.csv")
    sim_meta["seed_family_id"] = "simseed:"+sim_meta.seed.astype(str)
    sim_meta.to_csv(target/"synthetic_metadata.csv",index=False)
    inventory = [{k:r[k] for k in ("parent_track_id","source_group_id","label","split","file_sha256","history_hash","video_sha256")} for r in source["sources"]]
    pd.DataFrame(inventory).to_csv(target/"source_inventory.csv",index=False)
    root = Path(__file__).parent
    for name in ("trajectory_sequence.py","minirocket_inference.py","TRAJECTORY_SEQUENCE_V1.md","requirements-minirocket.txt"):
        shutil.copy2(root/name,target/name)
    code_dir = target/"reproducibility_source"
    code_dir.mkdir()
    for name in ("prepare_sequence_handoff.py","evaluate_sequence_handoff.py","package_sequence_handoff.py","sequence_simulator.py"):
        shutil.copy2(root/name,code_dir/name)
    figures(folder,evaluation,target)
    counts = []
    for name in ("train_real.npz","synthetic_train_experimental.npz","train_real_plus_synthetic.npz","validation.npz","test.npz"):
        data = load_arrays(target/name)
        counts.append(dict(file=name,windows=len(data["X"]),groups=len(set(data["group_id"])),
                           bird=int(sum(data["y"] == "bird")),drone=int(sum(data["y"] == "drone"))))
    contract = dict(version=CONTRACT_VERSION,fingerprint=SequenceConfig().fingerprint,config=asdict(SequenceConfig()),
        shape="(N,4,60)",dtype="float32",channels=list(CHANNELS),labels=["bird","drone"],
        score="Ridge margin: negative bird, nonnegative drone; NOT a probability",selected_arm=result["selected_arm"],
        real_source_video_group_overlap=0,sessions_verified=False,independent_test=False,
        group_policy="Same recording includes every object/derived window; never random-row split",
        synthetic_policy="Experimental only; current comparison does not support inclusion",
        train_files_are_alternative_arms_not_additional_splits=True,counts=counts)
    write_json(target/"input_contract.json",contract)
    write_json(target/"test_coverage.json",read_json(evaluation/"test_replay.json"))
    group_results = [f"| {arm} | {r['validation']['group']['macro_f1']:.4f} | {r['test']['group']['macro_f1']:.4f} | {r['test']['group']['accuracy']:.1%} | {r['test']['group']['recall']['drone']:.1%} |" for arm,r in result["arms"].items()]
    lines = ["# B → C: 2초 궤적 데이터 전달 및 검증 결과", "",
        "## 결론", "",
        "현재 전달 모델은 **실제 자료만 학습한 MiniRocket + Ridge**입니다. 이번 합성 데이터는 성능을 낮췄으므로 기본 학습에 포함하지 않습니다.",
        "카메라의 지하 위치 오류는 수정했지만, Sim-to-Real 및 합성 증강의 유효성을 입증한 결과는 아닙니다.", "",
        "## 데이터", "", "| 파일 | 창 수 | 영상/합성 비행 그룹 | bird | drone |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {c['file']} | {c['windows']} | {c['groups']} | {c['bird']} | {c['drone']} |" for c in counts]
    lines += ["", "창은 독립 영상 수가 아닙니다. 1초 stride로 인접한 2초 창이 겹칩니다. 두 train 파일은 서로 다른 실험 구성이지 서로 독립된 데이터가 아닙니다.",
        "실제 데이터는 train 33궤적/29영상그룹, validation 11궤적/10그룹, test 10궤적/10그룹입니다.",
        f"합성은 {preparation['synthetic_flights']}비행을 시도하여 {preparation['synthetic_usable_flights']}비행에서 {preparation['synthetic_windows']}개의 유효 창을 얻었습니다. 나머지도 attempts/rejections에 기록되어 있습니다.",
        "같은 번호의 괄호 파일은 같은 원본 영상 안의 다른 객체라는 사용자 확인을 반영했습니다. 기존 그룹/분할은 변경되지 않았고 영상 ID·파일명·확인 가능한 원본 해시에 따른 교차 분할 연결은 0건입니다.",
        f"**다른 원본 영상 사이의 촬영 세션 독립성은 확인되지 않았습니다.** train {audit['train_without_video_hash']}궤적의 원본 영상 해시는 여전히 없습니다. 원본 부재를 해결했다고 주장하지 않습니다.", "",
        "## 입력 계약", "", "`trajectory-sequence-1.0.1`, fingerprint `eb9be8154ee0f404`, `float32 (N,4,60)`.",
        "채널은 `q_x, q_y, d_x, d_y`. 2초·30Hz·1초 stride이며, bbox를 사용하지 않습니다.",
        "좌표 `(cx, cy*height/width)`를 중앙값으로 중심 이동하고 공통 반경으로 나눕니다. d는 초당 속도가 아니라 인접 q의 차이이며 첫 d=0입니다. 방향 회전 정규화·시간 늘리기는 하지 않습니다.",
        "원본 A의 stabilization.applied=True, timestamp_ms/frame_index 및 processed_width/height가 필요합니다. 누락 제한·저프레임 처리 등은 동봉 trajectory_sequence.py가 유일한 전처리 기준입니다.",
        "NPZ는 `X,y,group_id,sample_id`를 포함하며 train에는 `sample_weight,domain`도 있습니다. 메타데이터와 정답 y를 모델 입력 채널로 넣지 않습니다.", "",
        "## 모델 및 평가", "", "| 구성 | validation 영상 macro-F1 | test 영상 macro-F1 | test 영상 정확도 | test drone recall |", "|---|---:|---:|---:|---:|"]
    lines += group_results
    lines += ["", "위 표는 여러 객체의 점수를 평균한 **영상 단위**입니다. 시연의 실제 출력 단위인 객체별(track) 성능과 창별 성능은 evaluation_results.json에서 별도로 확인해야 합니다.",
        "MiniRocket의 약 9,996개 특징과 StandardScaler는 두 구성 모두 실제 train만으로 적합했습니다. 합성 데이터가 표현·스케일을 지배하지 않도록 제한한 비교이며, 가능한 모든 혼합 학습 방법을 검증한 것은 아닙니다.",
        f"Ridge alpha={result['alpha']}는 실제 train 그룹 3-fold CV에서 선택했습니다. 혼합 구성은 손실 가중치의 실제 75%/합성 25%, 클래스·원본 그룹별 균형 가중치입니다. 두 구성의 총 가중치는 같습니다.",
        "분류 임계값은 0으로 고정했습니다. 창 margin 평균 → 객체별 margin 평균 → 영상별 동일 객체 가중 평균을 사용했습니다. margin을 확률(%)로 해석하면 안 됩니다.",
        "validation 영상 macro-F1로 구성을 선택하고, 모델·선택 파일의 SHA-256을 고정한 뒤 test를 1.0.1로 변환했습니다. 이 test는 과거 이미 관찰한 개발 자료이므로 독립적인 새 test가 아닙니다.",
        "첫 float32 Ridge 실행에서 수치 경고가 나와, 설정을 그대로 두고 선형대수 계산만 float64로 바꿔 재실행했습니다. 첫 실행과 현재 실행의 모든 집계 성능지표가 동일했습니다. 원래 실행·test 열람 이력은 삭제하지 않았습니다.",
        f"합성 추가의 영상 macro-F1 차이(합성-실제) 그룹 bootstrap 95% 구간: {result['paired_bootstrap']['group_macro_f1_delta_q025_q50_q975'][0]:.3f} ~ {result['paired_bootstrap']['group_macro_f1_delta_q025_q50_q975'][2]:.3f}. 10영상뿐이며 세션 상관도 미확인이라 정밀한 모집단 추정이 아닙니다.", "",
        "## 카메라 수정 검증", "",
        "물리 엔진 v4는 유지하고 sequence runner를 1.4.0으로 올렸습니다. 새 실행 경로는 지상 1.5m 카메라를 사용하고, 거리·시선각·화각을 함께 계산합니다. 물리 궤적/특징을 사후 확대하지 않습니다.",
        "1.5m와 최대 시선각 55도, 최소 화각 5도는 설계 제약이지 실제 촬영 장비를 식별한 값이 아닙니다. 예전 결과 재현용 legacy 모드는 남아 있습니다.",
        f"train CV 평균 거리: legacy {preparation['cv']['legacy']['distance']:.4f} → ground {preparation['cv'][preparation['chosen']]['distance']:.4f}. 낮을수록 좋으므로 전체 개선이 아닙니다.",
        f"validation 거리: legacy {preparation['validation']['legacy']['distance']:.4f} → ground {preparation['validation'][preparation['chosen']]['distance']:.4f}. 새/드론 시계열 MMD도 함께 기록했습니다.",
        "더 크게 확대하는 span3 후보는 train CV에서 탈락했습니다. 지하 카메라 제거와 데이터 현실성 개선은 별개의 판단이며, 현재는 대량 합성 생성 승인 상태가 아닙니다.", "",
        "![평가 요약](evaluation_summary.png)", "", "![동일 2초 비교](trajectory_examples.png)", "",
        "## C에서 실행", "", "기존 서버 venv를 바꾸지 말고 별도 Python 3.12 환경에서 실행합니다.", "",
        "```powershell", "python -m venv .venv", ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-minirocket.txt",
        ".\\.venv\\Scripts\\python.exe minirocket_inference.py --model reference_model.joblib --input validation.npz",
        "# A에서 내보낸 단일 TrackSequence JSON으로 객체별 추론",
        ".\\.venv\\Scripts\\python.exe minirocket_inference.py --model reference_model.joblib --input your_track.json", "```", "",
        "신뢰할 수 있는 joblib 파일만 로드하세요. 모델은 코드 실행이 가능한 직렬화 형식입니다. SHA256SUMS.json으로 전달 무결성을 확인할 수 있습니다.",
        "재학습은 train_real을 기본으로 사용하고, 그룹을 유지한 CV를 사용합니다. 혼합 실험은 별도 train_real_plus_synthetic 및 그 sample_weight를 사용하세요. synthetic_metadata의 동일 seed_family_id도 하나의 계열로 유지하세요.",
        "이미 평가한 test로 추가 튜닝하지 마세요. 최종 외부 성능 주장은 별도로 새 촬영 세션에서 평가해야 합니다.", "",
        "## 남은 실제 제약", "",
        "1. 촬영 세션 확인 및 누락된 원본 영상 확보는 현재 파일만으로 끝낼 수 없습니다.",
        "2. 합성 데이터 추가가 이득이라는 근거를 확보하지 못했습니다. C의 우선 기준 모델은 실제-only로 유지합니다.",
        "3. B 전처리·C 모델의 서버 서비스 통합, 신뢰도 보류 정책 튜닝, 새 독립 영상 시연 검증은 이번 연구 실험과 별개입니다.", "",
        "## 구현 참고", "",
        "[aeon 공식 MiniRocket 예제](https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html)와 [MiniRocket 원 논문](https://arxiv.org/abs/2012.08791)의 구현을 사용했습니다. 코어 합성곱 변환을 자체 구현하지 않았습니다.", ""]
    (target/"README.md").write_text("\n".join(lines),encoding="utf-8")
    pd.DataFrame(counts).to_csv(target/"dataset_counts.csv",index=False)
    checksums = {str(p.relative_to(target)).replace("\\","/"):file_hash(p) for p in target.rglob("*") if p.is_file()}
    write_json(target/"SHA256SUMS.json",checksums)
    archive = folder/"C_handoff.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file():
                z.write(p,arcname=str(p.relative_to(folder)))
    write_json(folder/"handoff_artifact.json",dict(zip=archive.name,sha256=file_hash(archive),
                files=len(checksums)+1,contract=CONTRACT_VERSION,selected_arm=result["selected_arm"],
                synthetic_approved=False,independent_evaluation_ready=False))
    print(json.dumps(dict(archive=str(archive),counts=counts),indent=2))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--evaluation",default="model_evaluation_float64")
    args = parser.parse_args()
    package(args.folder,args.evaluation)
