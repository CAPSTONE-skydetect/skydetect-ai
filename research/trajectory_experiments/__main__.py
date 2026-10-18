from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.trajectory_experiments.manifest import (
    load_and_validate,
    prepare_legacy_manifest,
)
from research.trajectory_experiments.runner import (
    run_experiment,
    run_postprocessing,
    run_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible A trajectory experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--legacy-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--experiment-id", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--variants", nargs="*")
    run.add_argument("--dry-run", action="store_true")

    worker = commands.add_parser("worker")
    worker.add_argument("--job", type=Path, required=True)

    postprocess = commands.add_parser("postprocess")
    postprocess.add_argument("--manifest", type=Path, required=True)
    postprocess.add_argument("--experiment-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_legacy_manifest(
            args.legacy_root, args.output, experiment_id=args.experiment_id,
        )
    elif args.command == "validate":
        result = load_and_validate(args.manifest)
    elif args.command == "worker":
        result = run_worker(args.job)
    elif args.command == "postprocess":
        result = run_postprocessing(args.manifest, args.experiment_root)
    else:
        result = run_experiment(
            args.manifest, args.output, variants=args.variants, dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
