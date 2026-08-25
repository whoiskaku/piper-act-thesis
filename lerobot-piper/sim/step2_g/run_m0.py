#!/usr/bin/env python3
"""CLI for the offline Step 2 M0a role-separated synthetic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m0_synthetic import M0_LABEL, run_m0


def main() -> int:
    parser = argparse.ArgumentParser(description=M0_LABEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for the three immutable M0 JSON artifacts; existing targets are refused",
    )
    args = parser.parse_args()
    paths = run_m0(args.output_dir)
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    print(M0_LABEL)
    print(f"artifact: {paths['artifact']}")
    print(f"manifest: {paths['manifest']}")
    print(f"role-separated holdout q RMSE: {evaluation['q_rmse']:.3e}")
    print(f"role-separated holdout qdot RMSE: {evaluation['qdot_rmse']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
