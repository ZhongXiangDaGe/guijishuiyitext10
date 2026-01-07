#!/usr/bin/env python3
"""
verify_shards.py - 第三方验证器（工程增强版）

增强输出：
- chain_ok: 是否能从前一 shard state 推出 expected_chain_H 且与保存一致
- prev_proj_hash_ok: 保存的 prev_proj_hash 是否匹配我们计算的 prev_proj_hash
- positions_hash_ok: positions_hash 是否一致（若保存）
- det: 仍是符号位匹配率
"""

import argparse, csv, time, hashlib, random
import numpy as np
import torch
from torchvision import models

from utils import (
    safe_torch_load,
    weights_projection_from_state_dict, sha256_hex,
    derive_chain_hash, kdf_from_hash,
    normalize_positions, positions_hash_hex,
)

def build_model(model_name: str, num_classes: int, image_size: int):
    if model_name == "resnet18":
        m = models.resnet18(weights=None)
        if image_size <= 64:
            m.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.maxpool = torch.nn.Identity()
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "vit_b_16":
        return models.vit_b_16(weights=None, image_size=int(image_size), num_classes=num_classes)
    raise NotImplementedError("Only resnet18 and vit_b_16 supported")

def extract_bits_from_model(model, positions):
    bits=[]
    pd = dict(model.named_parameters())
    for name, idx in positions:
        p = pd.get(name, None)
        if p is None:
            bits.append(np.zeros(len(idx), dtype=np.uint8)); continue
        flat = p.view(-1).detach().cpu().numpy()
        bits.append((flat[idx] > 0).astype(np.uint8))
    return np.concatenate(bits) if len(bits)>0 else np.array([], dtype=np.uint8)

def _sorted_items(ckpts):
    items=[]
    if isinstance(ckpts, dict):
        for k, v in ckpts.items():
            try: nk = int(k)
            except: nk = k
            items.append((nk, str(k), v))
        try:
            items.sort(key=lambda x: int(x[0]))
        except:
            items.sort(key=lambda x: x[0])
    elif isinstance(ckpts, list):
        for i, v in enumerate(ckpts, start=1):
            items.append((i, str(i), v))
    else:
        raise RuntimeError("Unsupported checkpoints format")
    return items


def select_random_shards(saved_path, num_challenges, seed=0, allow_unsafe=False):
    data = safe_torch_load(saved_path, allow_unsafe=allow_unsafe)
    ckpts_serial = data.get("checkpoints", {})
    items = _sorted_items(ckpts_serial)
    shard_keys = [str(key_str) for _, key_str, _ in items]
    if not shard_keys:
        return []
    rng = random.Random(seed)
    if num_challenges >= len(shard_keys):
        return shard_keys
    return rng.sample(shard_keys, num_challenges)

def verify(saved_path, specific_shards=None, eta_G=0.85, out_csv=None, allow_unsafe=False):
    data = safe_torch_load(saved_path, allow_unsafe=allow_unsafe)
    ckpts_serial = data.get("checkpoints", {})
    mu = data.get("mu", None)
    id_P = data.get("id_P", None)
    meta = data.get("meta", {}) or {}
    cfg = data.get("config", {}) or {}
    model_name = meta.get("model", "resnet18")
    num_classes = int(meta.get("num_classes", 100))
    image_size = int(cfg.get("image_size", 32))

    items = _sorted_items(ckpts_serial)
    want = None
    if specific_shards:
        want = set([str(s) for s in specific_shards])

    rows = []
    for idx_num, key_str, entry in items:
        if want and str(key_str) not in want:
            continue

        state = entry.get("state", None)
        positions_raw = entry.get("positions", [])
        bits_raw = np.asarray(entry.get("bits", []), dtype=np.uint8)

        chain_H_saved = entry.get("chain_H", None)
        prev_proj_hash_saved = entry.get("prev_proj_hash", None)
        positions_hash_saved = entry.get("positions_hash", None)

        positions = normalize_positions(positions_raw)
        pos_hash = positions_hash_hex(positions) if positions else None

        # previous state
        prev_entry_state = None
        for jdx_num, _, jentry in items:
            if int(jdx_num) == int(idx_num) - 1:
                prev_entry_state = jentry.get("state", None)
                break

        expected_chain_H = None
        prev_proj_hash_expected = None
        chain_ok = None
        prev_proj_hash_ok = None

        if prev_entry_state is not None and mu is not None and id_P is not None:
            prev_proj_bytes = weights_projection_from_state_dict(prev_entry_state)
            prev_proj_hash_expected = sha256_hex(prev_proj_bytes)
            expected_chain_H = derive_chain_hash(prev_proj_bytes, int(idx_num), mu, id_P)
            if chain_H_saved is not None:
                chain_ok = (str(chain_H_saved) == str(expected_chain_H))
            if prev_proj_hash_saved is not None:
                prev_proj_hash_ok = (str(prev_proj_hash_saved) == str(prev_proj_hash_expected))
        elif prev_entry_state is None and int(idx_num) == 1:
            chain_ok = "n/a(genesis)"
            prev_proj_hash_ok = "n/a(genesis)"

        positions_hash_ok = None
        if positions_hash_saved is not None and pos_hash is not None:
            positions_hash_ok = (str(positions_hash_saved) == str(pos_hash))

        # pick chain for detection
        use_chain_H = chain_H_saved or expected_chain_H
        det = -1.0
        note = ""

        if use_chain_H is None:
            det = -1.0
            note = "no chain_H and cannot derive expected (missing prev shard?)"
        else:
            bits_expected, _ = kdf_from_hash(use_chain_H, n_bits=len(bits_raw), key_len=32)
            try:
                model = build_model(model_name, num_classes, image_size)
                model.load_state_dict(state, strict=False)
                model.eval()
                hat = extract_bits_from_model(model, positions)
                mlen = min(len(hat), len(bits_expected))
                det = float((hat[:mlen] == bits_expected[:mlen]).mean()) if mlen > 0 else 0.0
                if det < eta_G:
                    note = f"det_below_etaG({det:.3f}<{eta_G})"
            except Exception as e:
                det = -1.0
                note = f"load/inspect error: {e}"

        rows.append(
            {
                "shard": key_str,
                "det": det,
                "num_bits": int(len(bits_raw)),
                "chain_H_saved": chain_H_saved,
                "expected_chain_H": expected_chain_H,
                "chain_ok": chain_ok,
                "prev_proj_hash_saved": prev_proj_hash_saved,
                "prev_proj_hash_expected": prev_proj_hash_expected,
                "prev_proj_hash_ok": prev_proj_hash_ok,
                "positions_hash_saved": positions_hash_saved,
                "positions_hash": pos_hash,
                "positions_hash_ok": positions_hash_ok,
                "note": note,
            }
        )

    for r in rows:
        print(
            f"shard {r['shard']}: det={r['det']}, chain_ok={r['chain_ok']}, "
            f"prev_proj_hash_ok={r['prev_proj_hash_ok']}, pos_hash_ok={r['positions_hash_ok']}, note={r['note']}"
        )

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "shard","det","num_bits",
                    "chain_ok","chain_H_saved","expected_chain_H",
                    "prev_proj_hash_ok","prev_proj_hash_saved","prev_proj_hash_expected",
                    "positions_hash_ok","positions_hash_saved","positions_hash",
                    "note"
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r["shard"], r["det"], r["num_bits"],
                        r["chain_ok"], r["chain_H_saved"], r["expected_chain_H"],
                        r["prev_proj_hash_ok"], r["prev_proj_hash_saved"], r["prev_proj_hash_expected"],
                        r["positions_hash_ok"], r["positions_hash_saved"], r["positions_hash"],
                        r["note"],
                    ]
                )
        print(f"[info] written report to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saved", type=str, required=True)
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--eta_G", type=float, default=0.85)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--allow-unsafe-load", action="store_true")
    p.add_argument("--challenge", action="store_true", help="randomly sample shards for verification")
    p.add_argument("--num-challenges", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.challenge:
        sampled = select_random_shards(
            args.saved,
            num_challenges=int(args.num_challenges),
            seed=int(args.seed),
            allow_unsafe=args.allow_unsafe_load,
        )
        if not sampled:
            raise SystemExit("No shards found for challenge verification.")
        print(f"[info] challenge shards: {sampled}")
        verify(
            args.saved,
            specific_shards=[int(s) for s in sampled],
            eta_G=args.eta_G,
            out_csv=args.out,
            allow_unsafe=args.allow_unsafe_load,
        )
    else:
        verify(args.saved, specific_shards=args.shards, eta_G=args.eta_G, out_csv=args.out, allow_unsafe=args.allow_unsafe_load)
