from research.common_c_dataset import _class_label, audit_records


def _record(label: str, group: str, digest: str) -> dict[str, str]:
    return {"label": label, "source_group_id": group, "gt_sha256": digest}


def test_audit_blocks_too_few_groups_per_class() -> None:
    audit = audit_records([
        _record("bird", "b1", "1"),
        _record("drone", "d1", "2"),
        _record("drone", "d2", "3"),
        _record("drone", "d3", "4"),
    ])
    assert audit["status"] == "blocked"
    assert any("bird has 1" in reason for reason in audit["blockers"])


def test_audit_accepts_three_groups_per_class() -> None:
    records = [
        _record(label, f"{label}-{index}", f"{label}-{index}")
        for label in ("bird", "drone")
        for index in range(3)
    ]
    assert audit_records(records)["status"] == "ready"


def test_class_label_normalizes_cvat_object_names() -> None:
    assert _class_label("bird_2") == "bird"
