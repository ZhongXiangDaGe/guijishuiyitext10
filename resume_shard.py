#!/usr/bin/env python3
"""
resume_shard.py - shard 恢复/再训练（增强版）

增强点：
- --mode append|rewrite-from-shard
  append: 只更新指定 shard，不改后续 shards（可能导致链不一致，需明确标注）
  rewrite-from-shard: 更新 shard 后，后续 shards 标记 invalidated（或你可扩展为自动重算）
- 保存 lineage 信息到 top-level meta
"""

import argparse, copy, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models

from utils import (
    safe_torch_load, normalize_positions, normalize_target_map,
    compute_param_importance, apply_adaptive_dp_noise,
    weights_projection_from_state_dict, sha256_hex, derive_chain_hash,
)

def get_model(name="resnet18", num_classes=100):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise NotImplementedError("Only 'resnet18' supported")

def compute_watermark_loss(model, positions, target_map, embed_mode="hinge"):
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    if target_map is None:
        return loss
    param_dict = dict(model.named_parameters())
    for name, idx in positions:
        p = param_dict.get(name, None)
        if p is None:
            continue
        flat = p.view(-1)
        tm = target_map.get(name, None)
        if tm is None:
            continue
        sel = flat[idx].to(device)
        if isinstance(tm, tuple) and len(tm) == 2:
            signs_np, margin = tm
            signs_t = torch.from_numpy(np.asarray(signs_np, dtype=np.float32)).to(device)
            mlen = min(signs_t.numel(), sel.numel())
            signs_t = signs_t.view(-1)[:mlen]
            sel = sel.view(-1)[:mlen]
            viol = nn.functional.relu(torch.tensor(margin, device=device) - signs_t * sel)
            loss = loss + torch.mean(viol**2)
        else:
            arr = np.asarray(tm, dtype=np.float32)
            tgt = torch.from_numpy(arr).to(device)
            mlen = min(tgt.numel(), sel.numel())
            tgt = tgt.view(-1)[:mlen]
            sel = sel.view(-1)[:mlen]
            loss = loss + nn.functional.mse_loss(sel, tgt)
    return loss

def extract_bits_from_model(model, positions):
    bits = []
    param_dict = dict(model.named_parameters())
    for name, idx in positions:
        p = param_dict.get(name, None)
        if p is None:
            bits.append(np.zeros(len(idx), dtype=np.uint8))
            continue
        flat = p.view(-1).detach().cpu().numpy()
        vals = flat[idx]
        b = (vals > 0).astype(np.uint8)
        bits.append(b)
    return np.concatenate(bits) if len(bits) > 0 else np.array([], dtype=np.uint8)

def find_shard_entry(ckpts_raw, shard_idx: int):
    found_key = None
    found_entry = None
    if isinstance(ckpts_raw, dict):
        for k in list(ckpts_raw.keys()):
            if str(k) == str(shard_idx):
                found_key = k
                found_entry = ckpts_raw[k]
                break
    elif isinstance(ckpts_raw, list):
        idx = int(shard_idx) - 1
        if 0 <= idx < len(ckpts_raw):
            found_key = idx
            found_entry = ckpts_raw[idx]
    else:
        raise RuntimeError(f"Unsupported checkpoints storage type: {type(ckpts_raw)}")
    return found_key, found_entry

def resume_shard(saved_path, shard_idx, extra_epochs=40, lr=1e-4, lambda_w=1.5,
                 device="cuda", allow_unsafe_load=False, dp_params=None,
                 mode="append"):
    print(f"[info] Loading saved file: {saved_path}")
    data = safe_torch_load(saved_path, allow_unsafe=allow_unsafe_load)
    if "checkpoints" not in data:
        raise RuntimeError("Saved file doesn't contain 'checkpoints' field.")
    ckpts_raw = data["checkpoints"]

    meta = data.get("meta", {}) or {}
    model_name = meta.get("model", "resnet18")
    num_classes = int(meta.get("num_classes", 100))
    dataset = meta.get("dataset", "CIFAR100")
    mu = data.get("mu", None)
    id_P = data.get("id_P", None)

    found_key, found_entry = find_shard_entry(ckpts_raw, shard_idx)
    if found_entry is None:
        raise KeyError(f"Shard {shard_idx} not found in saved checkpoints.")

    state = found_entry.get("state", None)
    positions_raw = found_entry.get("positions", None)
    bits_raw = found_entry.get("bits", None)
    target_map_raw = found_entry.get("target_map", None)

    if state is None or positions_raw is None or bits_raw is None:
        raise RuntimeError("Checkpoint entry missing required fields ('state', 'positions', 'bits').")

    positions = normalize_positions(positions_raw)
    bits_arr = np.asarray(bits_raw, dtype=np.uint8)
    target_map = None
    if target_map_raw is not None:
        try:
            target_map = normalize_target_map(target_map_raw)
        except Exception:
            target_map = None

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device_t)
    model.load_state_dict(state, strict=False)
    model.train()

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    if dataset == "CIFAR10":
        train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    else:
        train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_Lw = compute_watermark_loss(model, positions, target_map, embed_mode="hinge").item()
    initial_Lw = abs(float(initial_Lw))
    print(f"[info] initial_Lw={initial_Lw:.6f}")

    consecutive_ok = 0
    last_eta = 0.0

    for epoch in range(extra_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_Lw = 0.0
        cnt = 0
        for step, (bx, by) in enumerate(train_loader):
            bx = bx.to(device_t); by = by.to(device_t)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss_task = criterion(out, by)
            Lw = compute_watermark_loss(model, positions, target_map, embed_mode="hinge")
            loss = loss_task + lambda_w * Lw
            loss.backward()
            optimizer.step()
            epoch_task_loss += float(loss_task.item())
            epoch_Lw += float(Lw.item())
            cnt += 1
            if step >= 200:
                break

        avg_task = epoch_task_loss / max(1, cnt)
        avg_Lw = epoch_Lw / max(1, cnt)
        hat = extract_bits_from_model(model, positions)
        mlen = min(len(hat), len(bits_arr))
        det_rate = float((hat[:mlen] == bits_arr[:mlen]).mean()) if mlen > 0 else 0.0
        last_eta = det_rate
        lw_decrease = (initial_Lw - avg_Lw) / initial_Lw if initial_Lw > 1e-12 else (0.0 if avg_Lw < 1e-6 else -1.0)
        print(f"[resume] epoch {epoch}: task_loss={avg_task:.4f}, avg_Lw={avg_Lw:.6f}, lw_decrease={lw_decrease:.4f}, eta={det_rate:.4f}")

        accept_epoch = (epoch + 1 >= 2) and (det_rate >= 0.85) and (lw_decrease >= 0.02)
        if accept_epoch:
            consecutive_ok += 1
            print(f"  pass embed cond ({consecutive_ok}/2)")
            if consecutive_ok >= 2:
                break
        else:
            consecutive_ok = 0

    print("[info] Applying Adaptive DP and saving updated shard checkpoint...")
    warm_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    importance_map = compute_param_importance(
        model, warm_loader, criterion, device_t,
        warmup_steps=dp_params.get("importance_warmup_steps", 50) if dp_params else 50
    )
    apply_adaptive_dp_noise(
        model, positions, importance_map,
        base_sigma=dp_params.get("DP_sigma", 1e-4) if dp_params else 1e-4,
        wm_ratio=dp_params.get("wm_ratio", 0.1) if dp_params else 0.1,
        importance_scale=dp_params.get("importance_scale", 2.0) if dp_params else 2.0,
    )

    entry_new = copy.deepcopy(found_entry)
    entry_new["state"] = copy.deepcopy(model.state_dict())
    entry_new["positions"] = [(n, idx.tolist()) for n, idx in positions]
    entry_new["bits"] = bits_arr.tolist()
    entry_new["target_map"] = target_map if target_map is not None else target_map_raw
    entry_new["notes"] = (entry_new.get("notes","") + f" | resumed@{time.strftime('%Y-%m-%d %H:%M:%S')}").strip()

    # chain consistency: recompute chain_H for this shard based on previous shard state if possible
    if mu is not None and id_P is not None:
        prev_state = None
        if isinstance(ckpts_raw, dict):
            prev_state = ckpts_raw.get(str(shard_idx - 1), {}).get("state", None) if shard_idx > 1 else None
        elif isinstance(ckpts_raw, list):
            if shard_idx > 1 and (shard_idx - 2) < len(ckpts_raw):
                prev_state = ckpts_raw[shard_idx - 2].get("state", None)

        if prev_state is not None:
            prev_proj = weights_projection_from_state_dict(prev_state)
            entry_new["prev_proj_hash"] = sha256_hex(prev_proj)
            entry_new["chain_H"] = derive_chain_hash(prev_proj, int(shard_idx), mu, id_P)

    # write back
    if isinstance(ckpts_raw, dict) and found_key is not None:
        ckpts_raw[found_key] = entry_new
    elif isinstance(ckpts_raw, list) and isinstance(found_key, int):
        ckpts_raw[found_key] = entry_new
    else:
        ckpts_raw[str(shard_idx)] = entry_new

    # mode handling: invalidate later shards if rewrite-from-shard
    invalidated = []
    if mode == "rewrite-from-shard":
        if isinstance(ckpts_raw, dict):
            for k in list(ckpts_raw.keys()):
                try:
                    if int(k) > int(shard_idx):
                        invalidated.append(str(k))
                        ckpts_raw[k]["notes"] = (ckpts_raw[k].get("notes","") + " | invalidated_by_resume").strip()
                except Exception:
                    continue
        elif isinstance(ckpts_raw, list):
            for i in range(shard_idx, len(ckpts_raw)):
                invalidated.append(str(i+1))
                if isinstance(ckpts_raw[i], dict):
                    ckpts_raw[i]["notes"] = (ckpts_raw[i].get("notes","") + " | invalidated_by_resume").strip()

    data["checkpoints"] = ckpts_raw
    lineage = data.get("lineage", {}) or {}
    lineage.update({
        "resumed_from": os.path.abspath(saved_path),
        "resumed_shard": int(shard_idx),
        "mode": mode,
        "invalidated_shards": invalidated,
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    })
    data["lineage"] = lineage

    save_new = saved_path.replace(".pth", f"_resumed_shard{shard_idx}_{mode}.pth")
    torch.save(data, save_new)
    print(f"[info] Saved to: {save_new}")
    return save_new

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saved", type=str, default="./outputs/po_lo_joint_adaptive.pth")
    p.add_argument("--shard", type=int, default=6)
    p.add_argument("--extra-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-w", type=float, default=1.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, default="append", choices=["append","rewrite-from-shard"])
    p.add_argument("--allow-unsafe-load", action="store_true")

    p.add_argument("--DP-sigma", type=float, default=1e-4)
    p.add_argument("--wm-ratio", type=float, default=0.1)
    p.add_argument("--importance-scale", type=float, default=2.0)
    p.add_argument("--importance-warmup-steps", type=int, default=50)
    args = p.parse_args()

    dp_params = {
        "DP_sigma": args.DP_sigma,
        "wm_ratio": args.wm_ratio,
        "importance_scale": args.importance_scale,
        "importance_warmup_steps": args.importance_warmup_steps,
    }
    resume_shard(
        args.saved, args.shard,
        extra_epochs=args.extra_epochs, lr=args.lr, lambda_w=args.lambda_w,
        device=args.device, allow_unsafe_load=args.allow_unsafe_load, dp_params=dp_params,
        mode=args.mode,
    )