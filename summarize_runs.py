#!/usr/bin/env python3
"""
Summarize one or more PoLO checkpoint files and optional robustness CSVs.

Example:
  python summarize_runs.py \
    --checkpoints ./outputs/po_lo_joint_adaptive.pth ./outputs/other_run.pth \
    --labels watermark_v1 baseline \
    --robustness-csv ./robustness_out/robustness_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import safe_torch_load


def _sorted_items(ckpts: Any) -> List[Tuple[int, str, Dict[str, Any]]]:
    items: List[Tuple[int, str, Dict[str, Any]]] = []
    if isinstance(ckpts, dict):
        for key, entry in ckpts.items():
            try:
                idx = int(key)
            except Exception:
                idx = len(items) + 1
            items.append((idx, str(key), entry))
    elif isinstance(ckpts, list):
        for idx, entry in enumerate(ckpts, start=1):
            items.append((idx, str(idx), entry))
    else:
        raise RuntimeError("Unsupported checkpoints format")

    items.sort(key=lambda item: item[0])
    return items


def _summarize_checkpoint(path: str, allow_unsafe: bool) -> Dict[str, Any]:
    data = safe_torch_load(path, allow_unsafe=allow_unsafe)
    ckpts = data.get("checkpoints", {})
    cfg = data.get("config", {}) or {}
    meta = data.get("meta", {}) or {}

    items = _sorted_items(ckpts)
    shard_rows = []
    accs: List[float] = []
    for _, shard_key, entry in items:
        acc = entry.get("test_acc", None)
        if isinstance(acc, (int, float)):
            accs.append(float(acc))
        shard_rows.append(
            {
                "shard": shard_key,
                "test_acc": acc,
                "bits_len": len(entry.get("bits", []) or []),
                "notes": entry.get("notes", ""),
            }
        )

    return {
        "path": path,
        "cfg": cfg,
        "meta": meta,
        "num_shards": len(items),
        "accs": accs,
        "shard_rows": shard_rows,
    }


def _summarize_robustness_csv(path: str) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    det_vals = [float(r["det_after"]) for r in rows if r.get("det_after")]
    acc_vals = [float(r["acc_after"]) for r in rows if r.get("acc_after")]
    return {
        "det_mean": sum(det_vals) / len(det_vals) if det_vals else float("nan"),
        "det_min": min(det_vals) if det_vals else float("nan"),
        "acc_mean": sum(acc_vals) / len(acc_vals) if acc_vals else float("nan"),
        "acc_min": min(acc_vals) if acc_vals else float("nan"),
    }


def _format_accs(accs: Iterable[float]) -> str:
    vals = list(accs)
    if not vals:
        return "n/a"
    mean = sum(vals) / len(vals)
    return f"mean={mean:.4f}, min={min(vals):.4f}, max={max(vals):.4f}"


def _print_checkpoint_summary(summary: Dict[str, Any], label: str) -> None:
    cfg = summary["cfg"]
    meta = summary["meta"]
    accs = summary["accs"]
    print(f"\n== {label} ==")
    print(f"path: {summary['path']}")
    print(
        "model: "
        f"{meta.get('model', cfg.get('model', 'n/a'))} | "
        f"dataset: {cfg.get('dataset', 'n/a')} | "
        f"shards: {summary['num_shards']} | "
        f"watermark: {cfg.get('enable_watermark', 'n/a')}"
    )
    print(f"acc: {_format_accs(accs)}")
    print("per-shard:")
    for row in summary["shard_rows"]:
        print(
            f"  shard {row['shard']}: "
            f"acc={row['test_acc']} | bits={row['bits_len']} | notes={row['notes']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="paths to .pth checkpoints")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels aligned with --checkpoints")
    ap.add_argument("--robustness-csv", nargs="*", default=None, help="optional robustness CSV files")
    ap.add_argument("--allow-unsafe-load", action="store_true", help="allow unsafe torch.load")
    args = ap.parse_args()

    labels = args.labels or []
    for idx, ckpt in enumerate(args.checkpoints):
        label = labels[idx] if idx < len(labels) else os.path.basename(ckpt)
        summary = _summarize_checkpoint(ckpt, allow_unsafe=args.allow_unsafe_load)
        _print_checkpoint_summary(summary, label)

    if args.robustness_csv:
        print("\n== robustness summary ==")
        for path in args.robustness_csv:
            stats = _summarize_robustness_csv(path)
            print(
                f"{path}: "
                f"det_mean={stats['det_mean']:.4f}, det_min={stats['det_min']:.4f}, "
                f"acc_mean={stats['acc_mean']:.4f}, acc_min={stats['acc_min']:.4f}"
            )


if __name__ == "__main__":
    main()
