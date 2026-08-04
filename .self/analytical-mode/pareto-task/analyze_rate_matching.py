#!/usr/bin/env python3
"""Classify PD returned/frontier points by prefill or decode capacity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PREFILL_FACTOR = 0.9
DECODE_FACTOR = 0.92


def add_capacity_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["prefill_capacity_seq_s"] = (
        data["(p)seq/s/worker"] * data["(p)workers"] * PREFILL_FACTOR
    )
    data["decode_capacity_seq_s"] = (
        data["(d)seq/s/worker"] * data["(d)workers"] * DECODE_FACTOR
    )
    data["capacity_ratio_prefill_decode"] = (
        data["prefill_capacity_seq_s"] / data["decode_capacity_seq_s"]
    )
    data["bottleneck"] = "decode"
    data.loc[
        data["prefill_capacity_seq_s"] < data["decode_capacity_seq_s"],
        "bottleneck",
    ] = "prefill"
    expected = data[["prefill_capacity_seq_s", "decode_capacity_seq_s"]].min(axis=1)
    data["expected_request_rate"] = expected
    data["request_rate_relative_error"] = (
        (data["request_rate"] - expected).abs() / expected.clip(lower=1e-12)
    )
    return data


def returned_points() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS.glob("*/pareto_df.csv")):
        data = pd.read_csv(path)
        model_key, system, mode = path.parent.name.split("__")
        data.insert(0, "mode", mode.upper())
        data.insert(0, "system", system)
        data.insert(0, "model_key", model_key)
        data.insert(0, "case_id", path.parent.name)
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def selected_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "model_key",
        "system",
        "mode",
        "bottleneck",
        "prefill_capacity_seq_s",
        "decode_capacity_seq_s",
        "capacity_ratio_prefill_decode",
        "expected_request_rate",
        "request_rate",
        "request_rate_relative_error",
        "ttft",
        "tpot",
        "tokens/s/user",
        "tokens/s/gpu",
        "tokens/s/gpu_cluster",
        "(p)bs",
        "(p)global_bs",
        "(p)workers",
        "(p)seq/s/worker",
        "(p)parallel",
        "(d)bs",
        "(d)global_bs",
        "(d)workers",
        "(d)seq/s/worker",
        "(d)parallel",
        "num_total_gpus",
    ]
    return data[[column for column in columns if column in data.columns]]


def main() -> None:
    returned = add_capacity_columns(returned_points())
    frontier = pd.read_csv(RESULTS / "pareto_frontiers.csv")
    frontier.insert(
        0,
        "case_id",
        frontier["model_key"] + "__" + frontier["system"] + "__" + frontier["mode"].str.lower(),
    )
    frontier = add_capacity_columns(frontier)

    selected_columns(returned).to_csv(RESULTS / "rate_matching_returned_points.csv", index=False)
    selected_columns(frontier).to_csv(RESULTS / "rate_matching_frontier_points.csv", index=False)

    summaries = []
    for scope, data in (("returned", returned), ("frontier", frontier)):
        grouped = (
            data.groupby(["model_key", "system", "mode", "bottleneck"])
            .size()
            .rename("points")
            .reset_index()
        )
        grouped.insert(0, "scope", scope)
        summaries.append(grouped)
    pd.concat(summaries, ignore_index=True).to_csv(
        RESULTS / "rate_matching_bottleneck_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
