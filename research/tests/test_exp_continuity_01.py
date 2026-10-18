from copy import deepcopy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from research.exp_continuity_01 import (
    DEFAULT_RUNTIME,
    PILOT,
    ROOT,
    audit_track_windows,
    canonical_start,
    load_models,
    prepare_import_paths,
    read_json,
    source_origins,
    write_csv,
    window_sample_origins,
)


CONFIG = SimpleNamespace(fps=30.0, samples=60, window_seconds=2.0)


def make_track():
    return {
        "history": [
            {
                "frame_index": frame,
                "timestamp_ms": frame * 500,
                "cx": 0.2 + frame * 0.01,
                "cy": 0.4,
            }
            for frame in range(9)
        ]
    }


def fake_windows(_track, _config):
    accepted = [
        {"relative_start_s": float(start), "window_start_s": float(start),
         "window_end_s": float(start + 2), "X": [start]}
        for start in (0, 1, 2)
    ]
    rejected = [{"relative_start_s": 3.0, "reason": "missing_fraction"}]
    return accepted, rejected


class ContinuityAuditTests(unittest.TestCase):
    def test_csv_writer_preserves_union_of_manifest_record_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.csv"
            write_csv(path, [{"record_type": "package", "split": "train"},
                             {"record_type": "paired", "gt_status": "CVAT"}])
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertIn("split", rows[0])
        self.assertIn("gt_status", rows[0])
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[1]["gt_status"], "CVAT")

    def test_source_origins_requires_ordered_unique_track_and_known_recoveries(self):
        track = make_track()
        origins, transitions = source_origins(track, {4, 5})

        self.assertEqual(origins, ["d0"] * 4 + ["yolo"] * 2 + ["d0"] * 3)
        self.assertEqual([(row["from_frame"], row["to_frame"]) for row in transitions],
                         [(3, 4), (5, 6)])
        with self.assertRaisesRegex(ValueError, "absent from Hybrid"):
            source_origins(track, {99})

    def test_resampling_support_includes_both_brackets_when_grid_sample_is_between_sources(self):
        track = make_track()
        origins = ["d0"] * 4 + ["yolo"] * 5

        # 1.9667s is bracketed by the final D0 point at 1.5s and first YOLO point at 2s.
        self.assertEqual(window_sample_origins(track, origins, 0.0), {"d0", "yolo"})
        self.assertEqual(window_sample_origins(track, origins, 2.0), {"yolo"})

    def test_audit_rejects_only_accepted_windows_whose_interpolation_support_crosses_boundary(self):
        track = make_track()
        before = deepcopy(track)
        audit = audit_track_windows(track, {4, 5, 6, 7, 8}, CONFIG, fake_windows)

        self.assertEqual([row["relative_start_s"] for row in audit["audited"]], [0, 1, 2, 3])
        self.assertEqual([row["relative_start_s"] for row in audit["audited"]
                          if row["crosses_source_boundary"]], [0.0, 1.0])
        self.assertEqual(sorted(float(key) for key in audit["segmented_by_start"]), [2.0])
        self.assertEqual(audit["audited"][-1]["segmented_rejection_reason"], "missing_fraction")
        self.assertEqual(track, before)

    def test_non_boundary_window_retains_c_preprocessor_rejection_reason(self):
        track = make_track()
        audit = audit_track_windows(track, set(), CONFIG, fake_windows)

        rejected = next(row for row in audit["audited"] if row["relative_start_s"] == 3.0)
        self.assertFalse(rejected["c_preprocess_accepted"])
        self.assertFalse(rejected["segmented_accepted"])
        self.assertEqual(rejected["continuity_rejection_reason"], "")
        self.assertEqual(rejected["segmented_rejection_reason"], "missing_fraction")

    def test_antialias_filter_is_conservatively_blocked_across_source_transitions(self):
        track = make_track()

        def filtered_windows(_track, _config):
            accepted = [{"relative_start_s": 2.0, "window_start_s": 2.0,
                         "window_end_s": 4.0, "anti_alias_applied": True}]
            return accepted, []

        audit = audit_track_windows(track, {4, 5, 6, 7, 8}, CONFIG, filtered_windows)

        row = audit["audited"][0]
        self.assertEqual(row["sources_used_for_resampling"], "yolo")
        self.assertTrue(row["filter_cross_boundary_risk"])
        self.assertEqual(row["continuity_rejection_reason"], "source_boundary_filter_support")
        self.assertEqual(audit["segmented_by_start"], {})

    def test_original_frame_gaps_and_timestamps_are_not_filled_or_rewritten(self):
        track = make_track()
        del track["history"][4]
        before_clock = [(p["frame_index"], p["timestamp_ms"]) for p in track["history"]]

        audit_track_windows(track, set(), CONFIG, fake_windows)

        after_clock = [(p["frame_index"], p["timestamp_ms"]) for p in track["history"]]
        self.assertEqual(after_clock, before_clock)
        self.assertNotIn(4, {frame for frame, _ in after_clock})

    def test_duplicate_or_unordered_frames_are_rejected_before_provenance_mapping(self):
        track = make_track()
        track["history"][1]["frame_index"] = 0

        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            source_origins(track, set())


class ContinuityPilotIntegrationTests(unittest.TestCase):
    def test_no_recovery_pilot_windows_and_frozen_scores_are_unchanged(self):
        pilot_path = PILOT / "b_inputs/drone_1_bird_1/a_track_sequence.json"
        baseline_path = DEFAULT_RUNTIME / "models/minirocket_classifier.joblib"
        stack_path = ROOT / "artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib"
        required = (pilot_path, baseline_path, stack_path)
        if not all(path.is_file() for path in required):
            self.skipTest("saved no-recovery pilot or frozen checkpoint is unavailable")

        prepare_import_paths(DEFAULT_RUNTIME)
        from ai_server.schemas import TrackSequence
        from research.trajectory_sequence import SequenceConfig, window_track

        payload = read_json(pilot_path)
        audit = audit_track_windows(payload, set(), SequenceConfig(), window_track)
        self.assertEqual(audit["transitions"], [])
        self.assertTrue(audit["accepted"])

        original_starts = [canonical_start(row["relative_start_s"])
                           for row in audit["accepted"]]
        segmented_starts = list(audit["segmented_by_start"])
        self.assertEqual(segmented_starts, original_starts)

        models = load_models(DEFAULT_RUNTIME, baseline_path, stack_path)
        track = TrackSequence.model_validate(payload)
        for model_name, bundle in models.items():
            with self.subTest(model=model_name):
                original_scores = bundle["score_fn"](audit["accepted"])
                segmented_windows = [audit["segmented_by_start"][start]
                                     for start in segmented_starts]
                segmented_scores = bundle["score_fn"](segmented_windows)
                self.assertEqual(len(segmented_windows), len(audit["accepted"]))
                self.assertEqual(original_scores.tolist(), segmented_scores.tolist())

                prediction = bundle["adapter"].predict(track)
                self.assertEqual(
                    [canonical_start(value) for value in prediction.window_starts_s],
                    original_starts,
                )
                self.assertEqual(prediction.window_scores, original_scores.tolist())


if __name__ == "__main__":
    unittest.main()
