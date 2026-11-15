"""Synthetic-only functional fixture and optional single end-to-end smoke."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def generate(folder):
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(12):
        split = "train" if i < 8 else "validation" if i < 10 else "test"
        label = ("bird", "drone")[i % 2]
        track = dict(track_id=i, source_video_id=f"fixture-{i}", processed_width=1280,
                     processed_height=720, stabilization={"applied": True, "method": "opencv_feature_cmc"},
                     history=[dict(frame_index=f, timestamp_ms=round(f*1000/30),
                                   cx=.25 + .002*f + .015*math.sin(f*(.03+.015*i)),
                                   cy=.45 + .018*math.cos(f*(.05+.004*i)),
                                   w=.02, h=.03, conf=.95) for f in range(94)])
        name = f"track-{i}.json"
        raw = json.dumps(track).encode()
        (folder / name).write_bytes(raw)
        records.append(dict(source_sha256=hashlib.sha256(f"synthetic-video-{i}".encode()).hexdigest(),
                            source_group=f"fixture-{i}", session_group="unknown", parent_id=f"parent-{i}",
                            object_id="object-1", segment_id="segment-1", label=label, split=split,
                            qa_disposition="accepted", track_path=name, track_sha256=hashlib.sha256(raw).hexdigest()))
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(dict(schema="stack-tracksequence-collection-1", records=records), indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    manifest = generate(a.folder)
    if a.smoke:
        from ai_server.services import sequence_model as seq
        from research import minirocket_stack_training as stack
        root = Path(__file__).resolve().parents[1]
        for module in (seq, stack):
            assert Path(module.__file__).resolve().is_relative_to(root)
        seq.N_KERNELS = 168  # Test-only override; never exposed by the real-data CLI.
        output = a.folder / "fit"
        sys.argv = ["stack", "train", "--collection", str(manifest), "--output", str(output)]
        stack.main()
        sys.argv = ["stack", "predict", "--model", str(output/"stack.joblib"), "--track", str(a.folder/"track-8.json")]
        stack.main()


if __name__ == "__main__":
    main()
