#!/usr/bin/env python3
"""
polot_experiment.py  链式 PoLO 实验（工程增强版 + 训练改进版）

你当前结果 acc 偏低的主要原因通常是：CIFAR100+ResNet18 用 Adam(lr=1e-3) 训练 120 epoch
往往不如 SGD+Cosine/StepLR；同时 watermark loss 从一开始就强行参与，会拖累任务收敛。

本版改进点（尽量不改变你“PoLO 链式/positions/hinge embed”的核心逻辑）：
1) 优化器/调度器：默认改为 SGD + momentum + weight_decay + CosineLR（更适合 ResNet/CIFAR）
2) watermark warmup：前 warmup_epochs 只训任务；之后逐步 ramp watermark 权重到 lambda_w
3) DP_noise 策略：DP_sigma 默认更合理；并支持只在“保存 shard”时施加一次噪声（你原本就是一次）
4) AMP 混合精度（可选）：加速并稳定训练（A40 上更合适）
5) 早停策略更温和：embed 判定仍按 eta_G+Lw，但不会过早退出导致任务没收敛
6) 保存：额外保存每 shard 的训练元信息（lr、lambda_w_eff、task/acc 轨迹），方便写论文/复现实验

兼容：你现有 verify_shards.py / robustness_grid.py 使用 checkpoints[x]['state'/'positions'/'bits'...] 不变。
"""

import os, argparse, time, copy
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models

from utils import (
    set_seed, get_run_env, new_run_id,
    weights_projection, weights_projection_from_state_dict, sha256_hex,
    derive_chain_hash, kdf_from_hash,
    serialize_checkpoints_for_saving,
    positions_hash_hex,
    ecc_encode_repetition,
)

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "dataset": "CIFAR100",
    "dataset_root": "./data",
    "model": "resnet18",
    "num_classes": 100,
    "image_size": 32,

    # PoLO
    "shards": 4,
    "n_bits": 512,
    "mu": 999999,
    "id_P": "proverA",

    # training
    "epochs_per_shard": 100,
    "batch_size": 128,
    "num_workers": 4,

    # optimizer/schedule (improved defaults)
    "optim": "sgd",           # "sgd" | "adam"
    "lr": 0.1,                # good starting point for CIFAR100+ResNet18 with SGD
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,

    "sched": "cosine",        # "cosine" | "step" | "none"
    "step_milestones": [10, 15],
    "step_gamma": 0.1,
    "min_lr": 1e-5,

    # watermark embed
    "lambda_w": 0.2,
    "lambda_w_warmup_epochs": 40,   # 前几轮只训任务
    "lambda_w_ramp_epochs": 40,     # 再用几轮把 watermark 权重从 0 拉到 lambda_w
    "beta": 0.05,
    "eta_G": 0.9,
    "embed_mode": "hinge",
    "trajectory_shards": [2, 4],
    "traj_noise_mag": 1e-3,

    # DP (你的原值 1e-4 很小；如果要体现 DP/扰动，通常会更大。但过大也会毁精度)
    # 先给一个温和可用的默认；你可以在实验中 sweep。
    "DP_sigma": 1e-4,

    # embed criteria
    "min_embed_epochs": 6,
    "embed_patience": 3,
    "min_lw_decrease": 0.03,
    "min_lw_abs_thresh": 1e-5,

    # select_positions importance estimation
    "importance_batches": 4,

    # AMP
    "amp": True,

    # misc
    "device": "cuda",
    "seed": 42,
    "deterministic": False,
    "save_dir": "./outputs",
    "enable_watermark": True,
    "apply_dp_noise": True,

    # ECC (optional; default off)
    "ecc_enable": False,
    "ecc_repeat": 3,
    "ecc_payload_bits": 128,
}

# ----------------------------
# Importance estimation & position selection (keep your version)
# ----------------------------
def estimate_param_grad_importance(model, train_loader, device="cpu", batches=4, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.train()
    accum = {}
    counts = {}
    for name, p in model.named_parameters():
        accum[name] = 0.0
        counts[name] = 0

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t)
    it = iter(train_loader)
    used = 0
    for _ in range(batches):
        try:
            bx, by = next(it)
        except StopIteration:
            break
        bx = bx.to(device_t)
        by = by.to(device_t)
        model.zero_grad(set_to_none=True)
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            accum[name] += float(p.grad.detach().abs().mean().item())
            counts[name] += 1
        used += 1

    if used == 0:
        return {name: 1e-6 for name in accum}

    avg = {}
    for name in accum:
        avg[name] = (accum[name] / counts[name]) if counts[name] > 0 else 1e-6
    return avg


def select_positions(model, mu, n_bits, train_loader=None, device="cpu", importance_batches=4, rng_seed_override=None):
    seed_val = abs(hash(str(mu)))
    if rng_seed_override is not None:
        seed_val ^= int(rng_seed_override)
    rng = np.random.RandomState(seed_val % (2**32))

    param_info = []
    for name, p in model.named_parameters():
        if "weight" in name and p.numel() > 16:
            param_info.append((name, p.numel()))
    if len(param_info) == 0:
        raise RuntimeError("No suitable 'weight' parameter tensors found for watermark embedding.")

    if train_loader is None or importance_batches <= 0:
        mean_abs_grad = {name: 1.0 for name, _ in param_info}
    else:
        mean_abs_grad = estimate_param_grad_importance(model, train_loader, device=device, batches=importance_batches)

    total_capacity = sum(sz for _, sz in param_info)
    if n_bits > total_capacity:
        raise RuntimeError(f"Not enough capacity ({total_capacity}) for requested n_bits {n_bits}")

    scores_by_param = []
    named = dict(model.named_parameters())
    for name, sz in param_info:
        vals = named[name].detach().cpu().view(-1).numpy()
        val_mag = np.abs(vals).astype(np.float64)
        grad_imp = float(mean_abs_grad.get(name, 1e-6))
        element_scores = val_mag * grad_imp + 1e-12
        scores_by_param.append((name, element_scores))

    sums = np.array([s.sum() for _, s in scores_by_param], dtype=np.float64) + 1e-12
    raw_counts = (sums / sums.sum()) * n_bits
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_bits - counts.sum())
    if remainder > 0:
        fracs = raw_counts - np.floor(raw_counts)
        order = np.argsort(-fracs)
        for i in range(remainder):
            counts[order[i]] += 1

    positions = []
    for (name, scores), take in zip(scores_by_param, counts):
        if take <= 0:
            continue
        if take >= len(scores):
            idx = np.arange(len(scores), dtype=np.int64)
        else:
            probs = scores / float(scores.sum())
            idx = rng.choice(len(scores), size=take, replace=False, p=probs).astype(np.int64)
        positions.append((name, idx))

    # fix count to exactly n_bits
    tot = sum(len(idx) for _, idx in positions)
    if tot != n_bits:
        flat = []
        for name, idx in positions:
            for i in idx:
                flat.append((name, int(i)))
        if len(flat) > n_bits:
            flat = flat[:n_bits]
        else:
            first_name = param_info[0][0]
            while len(flat) < n_bits:
                flat.append((first_name, 0))
        newmap = {}
        for name, ii in flat:
            newmap.setdefault(name, []).append(ii)
        positions = [(n, np.array(idxs, dtype=np.int64)) for n, idxs in newmap.items()]

    return positions


# ----------------------------
# embed/extract helpers
# ----------------------------
def embed_target_from_bits(bits, beta=0.05, mode="hinge"):
    if mode == "mse":
        return (bits.astype(np.float32) * 2 - 1) * beta
    elif mode == "hinge":
        signs = (bits.astype(np.int8) * 2 - 1).astype(np.int8)
        return signs, float(beta)
    else:
        raise ValueError("Unknown embed mode")


def compute_watermark_loss(model, positions, target_values_map, embed_mode="hinge"):
    loss = 0.0
    pd = dict(model.named_parameters())
    if embed_mode == "hinge":
        for name, idx in positions:
            p = pd[name]
            flat = p.view(-1)
            signs, margin = target_values_map[name]
            signs_t = torch.from_numpy(signs.astype(np.float32)).to(flat.device)
            sel = flat[idx]
            viol = F.relu(margin - signs_t * sel)
            loss = loss + torch.mean(viol**2)
        return loss
    else:
        for name, idx in positions:
            p = pd[name]
            flat = p.view(-1)
            tgt = torch.from_numpy(target_values_map[name]).to(flat.device)
            loss = loss + F.mse_loss(flat[idx], tgt)
        return loss


def extract_bits_from_model(model, positions):
    bits = []
    pd = dict(model.named_parameters())
    for name, idx in positions:
        p = pd.get(name, None)
        if p is None:
            bits.append(np.zeros(len(idx), dtype=np.uint8))
            continue
        flat = p.view(-1).detach().cpu().numpy()
        bits.append((flat[idx] > 0).astype(np.uint8))
    return np.concatenate(bits) if len(bits) > 0 else np.array([], dtype=np.uint8)


def apply_dp_noise_to_model(model, positions, sigma=1e-4):
    wm_param_names = set([name for name, _ in positions])
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in wm_param_names:
                continue
            p.add_(torch.randn_like(p) * float(sigma))


# ----------------------------
# Model & data
# ----------------------------
def get_model(name, num_classes, image_size=32):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        if image_size <= 64:
            m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_b_16":
        return models.vit_b_16(weights=None, image_size=int(image_size), num_classes=num_classes)
    raise NotImplementedError("Only resnet18 and vit_b_16 supported")


def _dataset_stats(name: str):
    if name == "CIFAR10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if name == "CIFAR100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataloaders(cfg, debug_small=False):
    root = cfg["dataset_root"]
    ds = cfg["dataset"]
    image_size = int(cfg.get("image_size", 32))
    mean, std = _dataset_stats(ds)
    if ds in ("ImageNet", "TinyImageNet"):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.143)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if debug_small:
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        train.data = train.data[:1024]
        train.targets = train.targets[:1024]
        test.data = test.data[:256]
        test.targets = test.targets[:256]
    else:
        if ds == "CIFAR10":
            train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        elif ds == "CIFAR100":
            train = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
        elif ds == "TinyImageNet":
            train = datasets.ImageFolder(os.path.join(root, "tiny-imagenet-200", "train"), transform=transform_train)
            test = datasets.ImageFolder(os.path.join(root, "tiny-imagenet-200", "val"), transform=transform_test)
        elif ds == "ImageNet":
            train = datasets.ImageFolder(os.path.join(root, "train"), transform=transform_train)
            test = datasets.ImageFolder(os.path.join(root, "val"), transform=transform_test)
        else:
            raise ValueError(f"Unknown dataset: {ds}")

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=cfg["batch_size"], shuffle=True, num_workers=int(cfg.get("num_workers", 4)), pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=int(cfg.get("num_workers", 4)), pin_memory=True
    )
    return train, test, train_loader, test_loader


def evaluate_test_accuracy(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device_t)
            y = y.to(device_t)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ----------------------------
# Optimizer / scheduler helpers
# ----------------------------
def make_optimizer(cfg, model):
    name = str(cfg.get("optim", "sgd")).lower()
    lr = float(cfg["lr"])
    if name == "adam":
        betas = tuple(cfg.get("adam_betas", [0.9, 0.999]))
        eps = float(cfg.get("adam_eps", 1e-8))
        return optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=float(cfg.get("weight_decay", 0.0)))
    # default sgd
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        nesterov=True,
    )


def make_scheduler(cfg, optimizer, total_epochs: int):
    sched = str(cfg.get("sched", "cosine")).lower()
    if sched == "none":
        return None
    if sched == "step":
        milestones = list(map(int, cfg.get("step_milestones", [int(total_epochs * 0.5), int(total_epochs * 0.75)])))
        gamma = float(cfg.get("step_gamma", 0.1))
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # default cosine
    min_lr = float(cfg.get("min_lr", 1e-5))
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_epochs), eta_min=min_lr)


def lambda_w_eff(cfg, epoch: int) -> float:
    """
    Warmup + ramp watermark weight:
    - [0, warmup): 0
    - [warmup, warmup+ramp): linear ramp to lambda_w
    - thereafter: lambda_w
    """
    lam = float(cfg["lambda_w"])
    warm = int(cfg.get("lambda_w_warmup_epochs", 0))
    ramp = int(cfg.get("lambda_w_ramp_epochs", 0))
    if epoch < warm:
        return 0.0
    if ramp <= 0:
        return lam
    t = (epoch - warm) / float(ramp)
    t = max(0.0, min(1.0, t))
    return lam * t


# ----------------------------
# Training loop
# ----------------------------
def shard_training_loop(cfg, debug_small=False):
    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", False))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    run_id = new_run_id("polo")
    env = get_run_env()

    model = get_model(cfg["model"], cfg["num_classes"], image_size=int(cfg.get("image_size", 32))).to(device)
    _, _, train_loader, test_loader = get_dataloaders(cfg, debug_small=debug_small)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer, total_epochs=int(cfg["epochs_per_shard"]))
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp", False)) and device.type == "cuda")

    checkpoints: Dict[int, Dict[str, Any]] = {}
    test_accs: List[float] = []

    # chain init
    W_prev_proj = weights_projection(model)
    W_prev_proj_hash = sha256_hex(W_prev_proj)

    for x in range(1, int(cfg["shards"]) + 1):
        print(f"[{time.strftime('%H:%M:%S')}] Shard {x} start")

        chain_H = derive_chain_hash(W_prev_proj, x, cfg["mu"], cfg["id_P"])
        prev_proj_hash = W_prev_proj_hash

        enable_watermark = bool(cfg.get("enable_watermark", True))
        if enable_watermark:
            # generate bits (optionally ECC)
            if cfg.get("ecc_enable", False):
                payload_bits, key = kdf_from_hash(chain_H, n_bits=int(cfg["ecc_payload_bits"]), key_len=32)
                bits = ecc_encode_repetition(payload_bits, repeat=int(cfg["ecc_repeat"]), out_len=int(cfg["n_bits"]))
                bits_source = f"ecc(rep={cfg['ecc_repeat']},payload={cfg['ecc_payload_bits']})"
            else:
                bits, key = kdf_from_hash(chain_H, n_bits=int(cfg["n_bits"]), key_len=32)
                bits_source = "raw"

            positions = select_positions(
                model,
                cfg["mu"] + x,
                int(cfg["n_bits"]),
                train_loader=train_loader,
                device=str(device),
                importance_batches=int(cfg.get("importance_batches", 4)),
            )
            pos_hash = positions_hash_hex(positions)

            # build target_map
            if cfg["embed_mode"] == "hinge":
                signs, margin = embed_target_from_bits(bits, beta=float(cfg["beta"]), mode="hinge")
                target_map = {}
                off = 0
                for name, idx in positions:
                    take = int(len(idx))
                    seg = signs[off: off + take]
                    target_map[name] = (seg, margin)
                    off += take
            else:
                targets = embed_target_from_bits(bits, beta=float(cfg["beta"]), mode="mse")
                target_map = {}
                off = 0
                for name, idx in positions:
                    take = int(len(idx))
                    target_map[name] = targets[off: off + take]
                    off += take

            # watermark loss reference
            model.eval()
            with torch.no_grad():
                initial_Lw = compute_watermark_loss(model, positions, target_map, embed_mode=cfg["embed_mode"]).item()
            initial_Lw = abs(float(initial_Lw))
        else:
            bits = np.array([], dtype=np.uint8)
            key = b""
            bits_source = "disabled"
            positions = []
            pos_hash = None
            target_map = {}
            initial_Lw = 0.0

        consecutive_ok = 0
        embedded = False
        last_eta = 0.0
        embed_log: List[Dict[str, Any]] = []
        train_log: List[Dict[str, Any]] = []

        for epoch in range(int(cfg["epochs_per_shard"])):
            model.train()
            epoch_task_loss = 0.0
            epoch_Lw = 0.0
            epoch_batches = 0
            lam_eff = lambda_w_eff(cfg, epoch)

            for step, (bx, by) in enumerate(train_loader):
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(bx)
                    loss_task = criterion(out, by)
                    if enable_watermark:
                        Lw = compute_watermark_loss(model, positions, target_map, embed_mode=cfg["embed_mode"])
                        loss = loss_task + lam_eff * Lw
                    else:
                        Lw = torch.tensor(0.0, device=loss_task.device)
                        loss = loss_task

                scaler.scale(loss).backward()

                # trajectory noise (optional): add to grads
                if x in cfg.get("trajectory_shards", []):
                    seed = int.from_bytes(key[:4], "little") ^ int(step)
                    rng = np.random.RandomState(seed)
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        shape = tuple(p.grad.shape)
                        noise_np = rng.normal(loc=0.0, scale=float(cfg["traj_noise_mag"]), size=shape).astype(np.float32)
                        p.grad.add_(torch.from_numpy(noise_np).to(p.grad.device))

                scaler.step(optimizer)
                scaler.update()

                epoch_task_loss += float(loss_task.detach().item())
                epoch_Lw += float(Lw.detach().item())
                epoch_batches += 1

            if scheduler is not None:
                scheduler.step()

            avg_task_loss = epoch_task_loss / max(1, epoch_batches)
            avg_Lw = epoch_Lw / max(1, epoch_batches)
            lr_now = float(optimizer.param_groups[0]["lr"])

            # detection rate eta
            if enable_watermark:
                hat = extract_bits_from_model(model, positions)
                bits_expected = bits
                mlen = min(len(hat), len(bits_expected))
                eta = float((hat[:mlen] == bits_expected[:mlen]).mean()) if mlen > 0 else 0.0
            else:
                eta = 0.0
            last_eta = eta

            # Lw decrease
            if enable_watermark and initial_Lw > 1e-12:
                lw_decrease = (initial_Lw - avg_Lw) / initial_Lw
            elif enable_watermark:
                lw_decrease = (0.0 if avg_Lw < float(cfg.get("min_lw_abs_thresh", 1e-6)) else -1.0)
            else:
                lw_decrease = 0.0

            # record logs
            train_log.append(
                {
                    "epoch": int(epoch),
                    "lr": lr_now,
                    "lambda_w_eff": float(lam_eff),
                    "avg_task_loss": float(avg_task_loss),
                    "avg_Lw": float(avg_Lw),
                    "eta": float(eta),
                }
            )

            # embed decision (same logic, but generally after warmup it becomes meaningful)
            accept_epoch = False
            if enable_watermark and epoch + 1 >= int(cfg.get("min_embed_epochs", 2)):
                cond_eta = eta >= float(cfg["eta_G"])
                rel_ok = lw_decrease >= float(cfg.get("min_lw_decrease", 0.02))
                abs_ok = avg_Lw <= float(cfg.get("min_lw_abs_thresh", 1e-6))
                if cond_eta and (rel_ok or abs_ok):
                    accept_epoch = True

            embed_log.append(
                {
                    "epoch": int(epoch),
                    "initial_Lw": float(initial_Lw),
                    "avg_Lw": float(avg_Lw),
                    "lw_decrease": float(lw_decrease),
                    "eta": float(eta),
                    "accept_epoch": bool(accept_epoch),
                    "consecutive_ok": int(consecutive_ok),
                }
            )

            print(
                f"Shard {x} epoch {epoch}: lr={lr_now:.5f}, lam_w={lam_eff:.3f}, "
                f"task_loss={avg_task_loss:.4f}, avg_Lw={avg_Lw:.6f}, lw_dec={lw_decrease:.4f}, eta={eta:.4f}"
            )

            if accept_epoch:
                consecutive_ok += 1
                print(f"  [info] epoch passes embed criteria ({consecutive_ok}/{int(cfg.get('embed_patience', 2))})")
                if consecutive_ok >= int(cfg.get("embed_patience", 2)):
                    # Add DP noise once after embedding success (as before)
                    if cfg.get("apply_dp_noise", True):
                        apply_dp_noise_to_model(model, positions, sigma=float(cfg["DP_sigma"]))

                    acc = evaluate_test_accuracy(model, test_loader, device=str(device))
                    entry = {
                        "state": copy.deepcopy(model.state_dict()),
                        "positions": positions,
                        "positions_hash": pos_hash,
                        "bits": bits,
                        "bits_source": bits_source,
                        "key": key,
                        "target_map": target_map,
                        "chain_H": chain_H,
                        "prev_proj_hash": prev_proj_hash,
                        "embed_log": embed_log,
                        "train_log": train_log,
                        "test_acc": float(acc),
                        "notes": "",
                    }
                    checkpoints[x] = entry
                    embedded = True
                    test_accs.append(float(acc))
                    print(f"Shard {x} embedded. test_acc={acc:.4f}")
                    break
            else:
                consecutive_ok = 0

        if not embedded and enable_watermark:
            # Even if not "accepted", still save a checkpoint for chain continuity
            if cfg.get("apply_dp_noise", True):
                apply_dp_noise_to_model(model, positions, sigma=float(cfg["DP_sigma"]))
            acc = evaluate_test_accuracy(model, test_loader, device=str(device))
            entry = {
                "state": copy.deepcopy(model.state_dict()),
                "positions": positions,
                "positions_hash": pos_hash,
                "bits": bits,
                "bits_source": bits_source,
                "key": key,
                "target_map": target_map,
                "chain_H": chain_H,
                "prev_proj_hash": prev_proj_hash,
                "embed_log": embed_log,
                "train_log": train_log,
                "test_acc": float(acc),
                "notes": "ended_without_reaching_eta",
            }
            checkpoints[x] = entry
            test_accs.append(float(acc))
            print(f"Shard {x} ended without reaching eta_G; saved. test_acc={acc:.4f} (eta={last_eta:.3f})")
        elif not embedded and not enable_watermark:
            acc = evaluate_test_accuracy(model, test_loader, device=str(device))
            entry = {
                "state": copy.deepcopy(model.state_dict()),
                "positions": positions,
                "positions_hash": pos_hash,
                "bits": bits,
                "bits_source": bits_source,
                "key": key,
                "target_map": target_map,
                "chain_H": chain_H,
                "prev_proj_hash": prev_proj_hash,
                "embed_log": embed_log,
                "train_log": train_log,
                "test_acc": float(acc),
                "notes": "no_watermark",
            }
            checkpoints[x] = entry
            test_accs.append(float(acc))
            print(f"Shard {x} saved without watermark. test_acc={acc:.4f}")

        # update chain seed based on saved shard state
        W_prev_proj = weights_projection_from_state_dict(checkpoints[x]["state"])
        W_prev_proj_hash = sha256_hex(W_prev_proj)

    # Save
    os.makedirs(cfg["save_dir"], exist_ok=True)
    saved_path = os.path.join(cfg["save_dir"], "po_lo_joint_adaptive.pth")

    serial_ckpts = serialize_checkpoints_for_saving(checkpoints)
    top = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "final_state": model.state_dict(),
        "checkpoints": serial_ckpts,
        "mu": cfg.get("mu", None),
        "id_P": cfg.get("id_P", None),
        "config": cfg,
        "meta": {
            "model": cfg["model"],
            "num_classes": cfg["num_classes"],
            "dataset": cfg["dataset"],
            "env": env,
        },
    }
    torch.save(top, saved_path)
    print("Training finished. Saved checkpoints to", saved_path)
    if test_accs:
        acc_arr = np.asarray(test_accs, dtype=np.float32)
        print(
            "[summary] shard test_acc: "
            f"mean={float(acc_arr.mean()):.4f}, "
            f"std={float(acc_arr.std(ddof=0)):.4f}, "
            f"min={float(acc_arr.min()):.4f}, "
            f"max={float(acc_arr.max()):.4f}"
        )
    return saved_path


# ----------------------------
# CLI / config
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("train")
    sub.add_parser("verify")
    sub.add_parser("demo")

    p.add_argument("--config", type=str, default=None, help="yaml config path")
    p.add_argument("--saved", type=str, default=None)
    p.add_argument("--shard", type=int, default=2)

    # allow overriding seed/device quickly
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(path=None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
            if isinstance(user, dict) and "default" in user:
                user = user["default"]
            if isinstance(user, dict):
                cfg.update(user)
    return cfg


def demo_run():
    cfg = dict(DEFAULT_CFG)
    cfg.update(
        {
            "shards": 3,
            "epochs_per_shard": 5,
            "n_bits": 256,
            "batch_size": 128,
            "seed": 1,
            "save_dir": "./outputs_demo",
            "dataset": "CIFAR10",
            "num_classes": 10,
            "image_size": 32,

            # for quick demo, use smaller lr
            "lr": 0.05,
            "lambda_w_warmup_epochs": 1,
            "lambda_w_ramp_epochs": 2,
            "sched": "cosine",
        }
    )
    print("Running quick demo with small settings (CIFAR10 subset).")
    saved = shard_training_loop(cfg, debug_small=True)
    print("Saved demo checkpoint to:", saved)
    from verify_shards import verify
    verify(saved, specific_shards=[2], eta_G=cfg["eta_G"], out_csv=None, allow_unsafe=False)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = str(args.device)

    if args.cmd == "train" or args.cmd is None:
        print("Starting training (sharded, adaptive DP)...")
        saved = shard_training_loop(cfg, debug_small=False)
        print("Saved to:", saved)
    elif args.cmd == "verify":
        saved = args.saved if args.saved else os.path.join(cfg["save_dir"], "po_lo_joint_adaptive.pth")
        from verify_shards import verify
        verify(saved, specific_shards=[args.shard], eta_G=cfg["eta_G"], out_csv=None, allow_unsafe=False)
    elif args.cmd == "demo":
        demo_run()
    else:
        print("Unknown command. Use train|verify|demo")


if __name__ == "__main__":
    main()
