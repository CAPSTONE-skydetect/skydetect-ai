from copy import deepcopy

import numpy as np
import pytest

from research.build_real_sequence_dataset import group_records
from research.prepare_sequence_handoff import audit_groups, camera_evidence
from research.sequence_simulator import AcquisitionProfile, latent_flight, observe_flight
from research.evaluate_sequence_handoff import aggregate_scores, balanced_group_weights, export_test, load_arrays, paired_interval
from research.minirocket_inference import decision_margins
from research.evaluate_wing_observation import paired_candidate, cv_gate
from research.refine_synthetic_selection import support_mask
from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig


def test_ground_camera_preserves_world_and_is_physical():
    flight = latent_flight("bird", "pigeon", 81002, duration=3)
    original = deepcopy(flight)
    for span in (.2, 2., 3.):
        profile = AcquisitionProfile(ground_camera=True, span_multiplier=span)
        record = observe_flight(flight, profile)
        camera = record["metadata"]["camera"]
        assert camera["position"][2] == pytest.approx(1.5)
        assert 5 <= camera["horizontal_fov_deg"] <= 70
        distance = np.linalg.norm(np.subtract(camera["look_at"], camera["position"]))
        assert distance == pytest.approx(record["metadata"]["camera_distance_m"])
        assert record["metadata"]["actual_elevation_deg"] <= 55.00001
        assert record["world_truth"] == original["world_truth"]
        assert np.isfinite(camera_evidence([record]).range_m).all()
    assert flight == original


def test_ground_camera_profiles_validate():
    with pytest.raises(ValueError, match="height"):
        AcquisitionProfile(ground_camera=True, camera_height_m=-1)
    with pytest.raises(ValueError, match="field"):
        AcquisitionProfile(ground_min_fov_deg=80)
    with pytest.raises(ValueError,match="Wing centroid"):
        AcquisitionProfile(wing_centroid_ratio=.3)


def test_wing_centroid_is_bounded_and_does_not_modify_physics_or_drone():
    base=AcquisitionProfile(ground_camera=True,dropout_rate=0,burst_probability=0,
                            drift_probability=0,camera_probability=0,jitter_px=0)
    shifted=paired_candidate(base)
    assert shifted.wing_centroid_ratio == .12
    for label,subtype in (("bird","pigeon"),("drone","consumer_quad")):
        flight=latent_flight(label,subtype,84,duration=3)
        a=observe_flight(flight,base)
        b=observe_flight(flight,shifted)
        assert a["world_truth"] == b["world_truth"]
        assert a["metadata"]["camera"] == b["metadata"]["camera"]
        assert [p["frame_index"] if p else None for p in a["optical_truth"]] == [p["frame_index"] if p else None for p in b["optical_truth"]]
        for old,new,row in zip(a["optical_truth"],b["optical_truth"],flight["world_truth"]):
            if old is None:
                continue
            delta=abs(new["cy"]-old["cy"])
            if label == "drone" or row["behavior"] == "glide":
                assert delta == 0
            else:
                assert delta <= .12*old["w"]*a["track"]["processed_width"]/a["track"]["processed_height"]+1e-12


def test_wing_gate_rejects_inconsistent_folds():
    import pandas as pd
    rows=[]
    for fold in range(3):
        rows.extend([dict(fold=fold,variant="control",distance=1.,usable=1.,objective=1.),
                     dict(fold=fold,variant="wing_centroid_0.12",distance=.9 if fold < 2 else 1.1,
                          usable=1.,objective=.9 if fold < 2 else 1.1)])
    assert not cv_gate(pd.DataFrame(rows))["passed"]


def test_synthetic_support_filter_uses_classwise_train_ranges():
    import pandas as pd
    real=pd.DataFrame([dict(label=label,group_id=f"{label}:{i}",
        screen_span=value,step_cv=value,band_3_10_ratio=value)
        for label,values in (("bird",(.1,.2,.3)),("drone",(.5,.6,.7)))
        for i,value in enumerate(values)])
    synthetic=pd.DataFrame([dict(label=label,group_id=f"s:{label}:{i}",
        sample_id=f"s:{label}:{i}:w0",screen_span=value,step_cv=value,band_3_10_ratio=value)
        for label,values in (("bird",(.2,.6)),("drone",(.6,.2)))
        for i,value in enumerate(values)])
    accepted, limits=support_mask(real,synthetic)
    assert accepted.tolist() == [True,False,True,False]
    assert limits["bird"]["screen_span"]["q95"] < limits["drone"]["screen_span"]["q05"]


def test_same_video_different_objects_remain_grouped(tmp_path):
    records = [dict(parent_track_id=f"t{i}", file_name=f"bird30({i}).json", label="bird",
                    source_video_id=f"upload{i}", center_hash=f"different{i}", video_sha256="",
                    split="train", group_review="unknown") for i in (1, 2)]
    group_records(records)
    assert records[0]["source_group_id"] == records[1]["source_group_id"]
    _, audit = audit_groups(dict(sources=records), tmp_path)
    assert not audit["cross_split_groups"]
    assert not audit["session_independence_verified"]
    records[1]["split"] = "validation"
    _, audit = audit_groups(dict(sources=records), tmp_path)
    assert audit["cross_split_groups"]


def test_video_hash_recovery_and_conflict(tmp_path):
    (tmp_path/"source.mp4").write_bytes(b"video fixture")
    records = [dict(parent_track_id="t", file_name="bird1.json", label="bird",
                    source_video_id="source", center_hash="c", video_sha256="",
                    split="train", group_review="unknown")]
    group_records(records)
    recovered, audit = audit_groups(dict(sources=records), tmp_path)
    assert audit["recovered_hash_tracks"] == ["t"]
    assert recovered[0]["video_sha256"]
    records[0]["video_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="hash changed"):
        audit_groups(dict(sources=records), tmp_path)


def test_group_weighting_and_equal_object_aggregation():
    y = np.array(["bird"]*4+["drone"]*2)
    group = np.array(["b1"]*3+["b2"]+["d1"]*2)
    weight = balanced_group_weights(y,group,6)
    assert weight[y == "bird"].sum() == pytest.approx(3)
    assert weight[group == "b1"].sum() == pytest.approx(weight[group == "b2"].sum())
    data = dict(y=y, group_id=group, sample_id=np.array(["a:w0","a:w1","b:w0","c:w0","d:w0","d:w1"]))
    _, tracks, groups = aggregate_scores(data,np.array([2,2,-2,-1,1,1]))
    assert len(tracks) == 4
    assert groups.loc[groups.group_id == "b1","decision"].iloc[0] == 0
    interval = paired_interval(groups,groups,draws=10)
    assert interval["group_macro_f1_delta_q025_q50_q975"] == [0,0,0]


def test_corrupt_displacement_rejected(tmp_path):
    x = np.zeros((2,4,60),np.float32)
    x[:,2,1] = 1
    np.savez(tmp_path/"bad.npz",X=x,y=np.array(["bird","drone"]),group_id=np.array(["b","d"]),sample_id=np.array(["b:w0","d:w0"]))
    with pytest.raises(ValueError,match="Displacement"):
        load_arrays(tmp_path/"bad.npz")


def test_test_export_requires_frozen_models_before_sources(tmp_path):
    (tmp_path/"selection.json").write_text('{}',encoding="utf-8")
    with pytest.raises(ValueError,match="frozen selection"):
        export_test({"sources":None},tmp_path/"test.npz",tmp_path/"selection.json")


def test_inference_checks_contract_before_using_model():
    with pytest.raises(ValueError,match="contract"):
        decision_margins(dict(contract_version="old",contract_id="old"),np.zeros((1,4,60),np.float32))
    model = dict(contract_version=CONTRACT_VERSION,contract_id=SequenceConfig().fingerprint,classes=["bird","drone"])
    assert decision_margins(model,np.zeros((0,4,60),np.float32)).size == 0
    with pytest.raises(ValueError,match="float32"):
        decision_margins(model,np.zeros((1,4,60),np.float64))


def test_real_minirocket_inference_round_trip():
    pytest.importorskip("aeon")
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from research.trajectory_sequence import normalize_window
    rng = np.random.default_rng(80)
    x = np.stack([normalize_window(np.cumsum(rng.normal(size=(60,2)),axis=0),SequenceConfig())[0] for _ in range(12)])
    y = np.array(["bird","drone"]*6)
    rocket = MiniRocket(n_kernels=84,random_state=80,n_jobs=1)
    scaler = StandardScaler(with_mean=False)
    z = scaler.fit_transform(rocket.fit_transform(x)).astype(np.float64)
    ridge = RidgeClassifier(alpha=1).fit(z,y)
    model = dict(rocket=rocket,scaler=scaler,classifier=ridge,classes=ridge.classes_.tolist(),
                 contract_version=CONTRACT_VERSION,contract_id=SequenceConfig().fingerprint)
    np.testing.assert_allclose(decision_margins(model,x),ridge.decision_function(z))
