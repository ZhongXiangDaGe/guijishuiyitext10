#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import pandas as pd


NUMERIC_COLS = [
    "det_before",
    "det_after",
    "ber_after",
    "acc_before",
    "acc_after",
    "acc_drop",
    "num_bits",
    "num_params",
    "sparsity",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _label_df(df: pd.DataFrame, csv_path: str, run_id: Optional[str]) -> pd.DataFrame:
    label = run_id if run_id else os.path.basename(csv_path)
    df = df.copy()
    df["run_id"] = label
    return df


def summarize(df: pd.DataFrame, include_run_summary: bool = True):
    df = _coerce_numeric(df)

    print("rows:", len(df))
    print(df.head(10).to_string(index=False))

    if "attack_type" not in df.columns:
        print("[error] missing attack_type column")
        return

    print("\nSummary by attack_type:")
    agg_cols = [c for c in ["det_after", "ber_after", "acc_after", "acc_drop", "sparsity"] if c in df.columns]
    print(df.groupby("attack_type")[agg_cols].agg(["mean", "std", "min", "count"]).to_string())

    if "attack_param" in df.columns:
        print("\nSummary by attack_type + attack_param:")
        g = df.groupby(["attack_type", "attack_param"])[agg_cols].agg(["mean", "std", "min", "count"])
        print(g.to_string())

    if include_run_summary and "run_id" in df.columns:
        print("\nRun-wise summary (mean/std):")
        run_cols = [c for c in ["det_after", "acc_after", "acc_drop"] if c in df.columns]
        print(df.groupby("run_id")[run_cols].agg(["mean", "std", "count"]).to_string())

    # worst cases (helpful for prune)
    if "det_after" in df.columns:
        worst = df.sort_values("det_after", ascending=True).head(15)
        print("\nWorst 15 rows by det_after:")
        cols = [
            c
            for c in [
                "run_id",
                "shard_key",
                "seed",
                "attack_type",
                "attack_param",
                "det_before",
                "det_after",
                "ber_after",
                "acc_before",
                "acc_after",
                "acc_drop",
                "sparsity",
                "notes",
            ]
            if c in df.columns
        ]
        print(worst[cols].to_string(index=False))

    # shard-wise
    if "shard_key" in df.columns and "det_after" in df.columns:
        print("\nShard-wise det_after (mean/min):")
        print(df.groupby("shard_key")[["det_after"]].agg(["mean", "min", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, nargs="+", default=["./robustness_out/robustness_grid.csv"])
    ap.add_argument("--run-id", type=str, nargs="*", default=None, help="optional labels aligned with --csv")
    args = ap.parse_args()

    run_ids = args.run_id if args.run_id else []
    frames = []
    for idx, path in enumerate(args.csv):
        df = pd.read_csv(path)
        label = run_ids[idx] if idx < len(run_ids) else None
        frames.append(_label_df(df, path, label))

    merged = pd.concat(frames, ignore_index=True)
    summarize(merged, include_run_summary=True)


if __name__ == "__main__":
    main()
