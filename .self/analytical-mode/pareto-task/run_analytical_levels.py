#!/usr/bin/env python3
"""Run SILICON and three ANALYTICAL estimate levels in an isolated result set."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import run_pareto as base


ROOT = Path(__file__).resolve().parent / "analytical-levels"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
LEVELS = ["low", "standard", "high"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=base.MODELS, default=list(base.MODELS))
    parser.add_argument("--systems", nargs="+", choices=base.SYSTEMS, default=base.SYSTEMS)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    base.RESULTS = RESULTS
    base.LOGS = LOGS

    statuses = []
    for model_key in args.models:
        for system in args.systems:
            cases = [("SILICON", "standard"), *[("ANALYTICAL", level) for level in args.levels]]
            for mode, level in cases:
                logging.info("Running %s / %s / %s / %s", model_key, system, mode, level)
                status = base.run_case(
                    model_key,
                    system,
                    mode,
                    args.force,
                    analytical_level=level,
                    label_analytical_level=True,
                )
                statuses.append(status)
                pd.DataFrame(statuses).to_csv(RESULTS / "run_summary.csv", index=False)
                logging.info("Completed %s: %s", status["case_id"], status["status"])

    (RESULTS / "run_summary.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
