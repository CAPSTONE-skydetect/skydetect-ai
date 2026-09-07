"""Import labeled A tracks through a reviewed, explicit acquisition manifest."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS, FeatureConfig, extract_features
from .io import write_json


def load_tracks(path):
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(root, dict) and "history" in root:
        return [root]
    if isinstance(root, dict) and "tracks" in root:
        tracks = root["tracks"]
    elif isinstance(root, list):
        tracks = root
    else:
        raise ValueError("Expected TrackSequence, list[TrackSequence], or {tracks: [...]}")
    if not tracks or not all(isinstance(t, dict) and "history" in t for t in tracks):
        raise ValueError("Point-only lists lack track provenance; wrap them in TrackSequence")
    return tracks


class RealTrackVerifier:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()

    def import_manifest(self, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        entries = manifest.get("videos", [])
        if not entries:
            raise ValueError("Manifest requires a nonempty videos list")
        rows, diagnostics = [], []
        ids = set()
        for entry in entries:
            if entry["label"] not in ("bird", "drone") or entry["split"] not in ("calibration", "test"):
                raise ValueError("Real labels must be bird/drone; splits calibration/test")
            group = str(entry["source_group_id"]).strip()
            if not group:
                raise ValueError("source_group_id must identify the ORIGINAL source video/session")
            if entry.get("coordinate_space") != "post_cmc_residual_observation":
                raise ValueError("This comparison requires an explicitly declared post-CMC coordinate space")
            path = manifest_path.parent / entry["track_path"]
            for track in load_tracks(path):
                stabilization = track.get("stabilization", {})
                if stabilization.get("applied") is False:
                    raise ValueError("Raw, unstabilized A tracks cannot be mixed with the post-CMC simulator")
                sample_id = f"real:{group}:{track['source_video_id']}:{track['track_id']}"
                if sample_id in ids:
                    raise ValueError("Duplicate track/video identity")
                ids.add(sample_id)
                history = track["history"]
                result = extract_features(history, entry["frame_width"], entry["frame_height"],
                                          entry.get("fps"), self.config)
                row = dict(sample_id=sample_id, family_id=f"real:{group}", label=entry["label"],
                           subtype=entry.get("subtype", "unknown"), split=entry["split"],
                           feature_status=result["feature_status"], feature_version=result["feature_version"],
                           feature_config_id=result["feature_config_id"],
                           rejection_reason=";".join(result["reasons"]),
                           review_status=entry.get("review_status", "unreviewed"),
                           video_conditions=entry.get("conditions", "unspecified"),
                           a_mean_conf=track.get("quality", {}).get("mean_conf"),
                           a_missing_ratio=track.get("quality", {}).get("missing_ratio"),
                           **{k: (result["features"] or {}).get(k) for k in FEATURE_COLUMNS},
                           **result["quality"])
                rows.append(row)
                diagnostics.append(dict(sample_id=sample_id, source_path=str(path.resolve()),
                                        review_status=row["review_status"], feature_result=result,
                                        acquisition=entry, a_quality=track.get("quality")))
        table = pd.DataFrame(rows)
        if (table.groupby("family_id").split.nunique() > 1).any():
            raise ValueError("Original video/session appears in both calibration and test")
        return table, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, help="Synthetic manifest.json, to share feature settings")
    args = parser.parse_args()
    config = FeatureConfig(**json.loads(args.feature_config.read_text(encoding="utf-8"))["feature_config"]) if args.feature_config else FeatureConfig()
    table, diagnostics = RealTrackVerifier(config).import_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "real_features_v3.csv", index=False)
    write_json(args.output / "real_import_report.json", diagnostics)
    print(f"Imported {len(table)} tracks. Feature acceptance is not manual tracking validation.")


if __name__ == "__main__":
    main()
