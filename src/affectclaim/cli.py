"""Command-line interface for reproducible AffectClaim runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from affectclaim.adapters.manifest import ManifestAdapter
from affectclaim.config import AppConfig
from affectclaim.factory import build_manifest_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="affectclaim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the manifest-based baseline")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path("configs/default.json"))
    run.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        raise AssertionError(f"Unhandled command: {args.command}")
    adapter = ManifestAdapter.from_json(args.manifest)
    pipeline = build_manifest_pipeline(adapter, AppConfig.from_json(args.config))
    results = []
    for sample in adapter.samples():
        for record in pipeline.analyse(sample):
            results.append(record.to_dict())
            print(f"[{record.action.value}] {record.realised_text}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
