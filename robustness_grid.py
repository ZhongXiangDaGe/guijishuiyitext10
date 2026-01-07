#!/usr/bin/env python3
"""
robustness_grid.py - PoLO checkpoints 鲁棒性评测（增强版）
"""

import os, argparse, time, csv
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models

from utils import safe_torch_load


def build_model(name="resnet18", num_classes=100, image_size=32):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        if image_size <= 64:
            m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_b_16":
        return models.vit_b_16(weights=None, image_size=int(image_size), num_classes=num_classes)
    raise NotImplementedError("Only 'resnet18' and 'vit_b_16' supported")


def accuracy_on_loader(model, loader, device="cuda", max_batches: Optional[int] = None):
    model.eval()
    correct = 0
    total = 0
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x = x.to(device_t)
            y = y.to(device_t)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if max_batches is not None and (bi + 1) >= max_batches:
                break
    return correct / total if total > 0 else 0.0


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


def det_and_ber(model, positions, bits_expected: np.ndarray) -> Tuple[float, float]:
    hat = extract_bits_from_model(model, positions)
    mlen = min(len(hat), len(bits_expected))
    if mlen <= 0:
        return 0.0, 1.0
    eq = (hat[:mlen] == bits_expected[:mlen])
    det = float(eq.mean())
    ber = float((~eq).mean())
    return det, ber


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_sparsity(model) -> float:
    zeros = 0
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach()
            zeros += int((t == 0).sum().item())
            total += t.numel()
    return zeros / total if total > 0 else 0.0


# ---- attacks ----
def _dataset_stats(name: str):
    if name == "CIFAR10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if name == "CIFAR100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def attack_finetune(
    state_dict,
    device,
    dataset_root,
    epochs=3,
    lr=1e-4,
    batch_size=128,
    num_classes=100,
    seed=0,
    model_name="resnet18",
    dataset="CIFAR100",
    image_size=32,
    max_steps_per_epoch=50,
):
    torch.manual_seed(seed)
    model = build_model(model_name, num_classes, image_size=image_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.train()

    mean, std = _dataset_stats(dataset)
    if dataset in ("ImageNet", "TinyImageNet"):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        if dataset == "TinyImageNet":
            train = datasets.ImageFolder(os.path.join(dataset_root, "tiny-imagenet-200", "train"), transform=transform)
        else:
            train = datasets.ImageFolder(os.path.join(dataset_root, "train"), transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        if dataset == "CIFAR10":
            train = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
        else:
            train = datasets.CIFAR100(root=dataset_root, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _e in range(epochs):
        for i, (bx, by) in enumerate(loader):
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            if i + 1 >= max_steps_per_epoch:
                break
    return model


def attack_prune(
    state_dict,
    device,
    amount=0.3,
    num_classes=100,
    model_name="resnet18",
    prune_mode="global_unstructured",
    image_size=32,
):
    import torch.nn.utils.prune as prune

    model = build_model(model_name, num_classes, image_size=image_size).to(device)
    model.load_state_dict(state_dict, strict=False)

    if prune_mode == "global_unstructured":
        parameters_to_prune = []
        for module in model.modules():
            if hasattr(module, "weight") and isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        for module, _ in parameters_to_prune:
            try:
                prune.remove(module, "weight")
            except Exception:
                pass

    elif prune_mode == "layerwise_unstructured":
        for module in model.modules():
            if hasattr(module, "weight") and isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=amount)
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass

    elif prune_mode == "structured_ln":
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass
            elif isinstance(module, nn.Linear):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass
    else:
        raise ValueError(f"Unknown prune_mode: {prune_mode}")

    return model


def attack_noise(state_dict, device, sigma=1e-3, seed=0, num_classes=100, model_name="resnet18", image_size=32):
    torch.manual_seed(seed)
    model = build_model(model_name, num_classes, image_size=image_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.randn_like(p) * sigma)
    return model


def quantize_tensor(t: torch.Tensor, n_bits: int):
    if n_bits >= 32:
        return t.clone()
    qmin = -(2 ** (n_bits - 1))
    qmax = (2 ** (n_bits - 1) - 1)
    max_val = t.abs().max()
    if float(max_val) == 0.0:
        return t.clone()
    scale = max_val / qmax
    q = (t / scale).round().clamp(qmin, qmax)
    return q * scale


def attack_quantize(state_dict, device, n_bits=8, num_classes=100, model_name="resnet18", image_size=32):
    model = build_model(model_name, num_classes, image_size=image_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.copy_(quantize_tensor(p, n_bits))
    return model


def iter_shards(ckpts_obj):
    if isinstance(ckpts_obj, dict):
        items = []
        for k, v in ckpts_obj.items():
            try:
                nk = int(k)
            except Exception:
                nk = k
            items.append((nk, str(k), v))
        try:
            items.sort(key=lambda x: int(x[0]))
        except Exception:
            items.sort(key=lambda x: x[0])
        for _, keystr, v in items:
            yield keystr, v
    elif isinstance(ckpts_obj, list):
        for i, v in enumerate(ckpts_obj):
            yield str(i + 1), v
    else:
        raise RuntimeError("Unsupported checkpoints format")


def choose_prune_amount_keep_acc(state, device_t, test_loader, acc_before, delta, grid, num_classes, model_name, prune_mode):
    best = None
    for amt in sorted(grid):
        model_p = attack_prune(state, device_t, amount=amt, num_classes=num_classes, model_name=model_name, prune_mode=prune_mode)
        acc = accuracy_on_loader(model_p, test_loader, device_t, max_batches=None)
        if acc >= (acc_before - delta):
            best = (amt, acc, measure_sparsity(model_p))
        else:
            break
    return best


def run_grid(
    saved_path: str,
    outdir: str,
    device: str,
    seeds,
    finetune_epochs,
    prune_amounts,
    noise_sigmas,
    quant_bits,
    prune_modes,
    prune_keep_acc_delta: Optional[float] = None,
    prune_keep_grid: Optional[List[float]] = None,
    dataset_root="./data",
    allow_unsafe_load: bool = False,  # FIX
):
    os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, "robustness_grid.csv")

    header = [
        "timestamp",
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
        "num_bits",
        "num_params",
        "sparsity",
        "notes",
    ]
    baseline_rows = []
    with open(outcsv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        print(f"[info] loading saved checkpoints from: {saved_path}")
        data = safe_torch_load(saved_path, allow_unsafe=allow_unsafe_load)  # FIX
        if "checkpoints" not in data:
            raise RuntimeError("Saved checkpoint file has no 'checkpoints' field.")

        ckpts = data["checkpoints"]
        meta = data.get("meta", {}) or {}
        cfg = data.get("config", {}) or {}
        model_name = meta.get("model", "resnet18")
        dataset = meta.get("dataset", "CIFAR100")
        num_classes = int(meta.get("num_classes", 100))
        image_size = int(cfg.get("image_size", 32))

        mean, std = _dataset_stats(dataset)
        if dataset in ("ImageNet", "TinyImageNet"):
            test_transform = transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.143)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            if dataset == "TinyImageNet":
                testset = datasets.ImageFolder(os.path.join(dataset_root, "tiny-imagenet-200", "val"), transform=test_transform)
            else:
                testset = datasets.ImageFolder(os.path.join(dataset_root, "val"), transform=test_transform)
        else:
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            if dataset == "CIFAR10":
                testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=test_transform)
            else:
                testset = datasets.CIFAR100(root=dataset_root, train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
        device_t = torch.device(device if torch.cuda.is_available() else "cpu")

        for shard_key, entry in iter_shards(ckpts):
            print(f"\n=== shard {shard_key} ===")
            state = entry.get("state", None)
            positions_raw = entry.get("positions", None)
            bits_raw = entry.get("bits", None)
            if state is None or positions_raw is None or bits_raw is None:
                print(f"[warn] shard {shard_key} missing required fields; skipping")
                continue

            positions = [(name, np.asarray(idxs, dtype=np.int64)) for name, idxs in positions_raw]
            bits_arr = np.asarray(bits_raw, dtype=np.uint8)

            model_base = build_model(model_name, num_classes, image_size=image_size).to(device_t)
            model_base.load_state_dict(state, strict=False)
            model_base.eval()

            det_before, _ = det_and_ber(model_base, positions, bits_arr)
            acc_before = accuracy_on_loader(model_base, test_loader, device_t)
            nparams = count_params(model_base)
            sparsity0 = measure_sparsity(model_base)

            print(f"[info] baseline: det={det_before:.4f}, acc={acc_before:.4f}, sparsity={sparsity0:.4f}")
            baseline_rows.append((float(det_before), float(acc_before), float(sparsity0)))
            writer.writerow(
                [
                    time.time(),
                    shard_key,
                    "",
                    "none",
                    "",
                    f"{det_before:.4f}",
                    f"{det_before:.4f}",
                    f"{0.0:.4f}",
                    f"{acc_before:.4f}",
                    f"{acc_before:.4f}",
                    f"{0.0:.4f}",
                    int(len(bits_arr)),
                    int(nparams),
                    f"{sparsity0:.6f}",
                    "baseline",
                ]
            )
            f.flush()

            for seed in seeds:
                # finetune
                for ep in finetune_epochs:
                    model_after = attack_finetune(
                        state,
                        device_t,
                        dataset_root=dataset_root,
                        epochs=ep,
                        lr=1e-4,
                        batch_size=128,
                        num_classes=num_classes,
                        seed=seed,
                        model_name=model_name,
                        dataset=dataset,
                        image_size=image_size,
                    )
                    det_after, ber = det_and_ber(model_after, positions, bits_arr)
                    acc_after = accuracy_on_loader(model_after, test_loader, device_t)
                    writer.writerow(
                        [
                            time.time(),
                            shard_key,
                            seed,
                            "finetune",
                            f"epochs={ep}",
                            f"{det_before:.4f}",
                            f"{det_after:.4f}",
                            f"{ber:.4f}",
                            f"{acc_before:.4f}",
                            f"{acc_after:.4f}",
                            f"{(acc_before-acc_after):.4f}",
                            int(len(bits_arr)),
                            int(nparams),
                            f"{measure_sparsity(model_after):.6f}",
                            "finetune",
                        ]
                    )
                    f.flush()

                # prune
                for mode in prune_modes:
                    if prune_keep_acc_delta is not None and prune_keep_grid is not None:
                        best = choose_prune_amount_keep_acc(
                            state,
                            device_t,
                            test_loader,
                            acc_before,
                            prune_keep_acc_delta,
                            prune_keep_grid,
                            num_classes,
                            model_name,
                            mode,
                        )
                        if best is None:
                            writer.writerow(
                                [
                                    time.time(),
                                    shard_key,
                                    seed,
                                    "prune_keepacc",
                                    f"mode={mode}",
                                    f"{det_before:.4f}",
                                    f"{0.0:.4f}",
                                    f"{1.0:.4f}",
                                    f"{acc_before:.4f}",
                                    f"{0.0:.4f}",
                                    f"{(acc_before-0.0):.4f}",
                                    int(len(bits_arr)),
                                    int(nparams),
                                    f"{0.0:.6f}",
                                    f"no_amount_keeps_acc(delta={prune_keep_acc_delta})",
                                ]
                            )
                            f.flush()
                        else:
                            amt, acc_sel, sparsity_sel = best
                            model_after = attack_prune(
                                state,
                                device_t,
                                amount=amt,
                                num_classes=num_classes,
                                model_name=model_name,
                                prune_mode=mode,
                                image_size=image_size,
                            )
                            det_after, ber = det_and_ber(model_after, positions, bits_arr)
                            writer.writerow(
                                [
                                    time.time(),
                                    shard_key,
                                    seed,
                                    "prune_keepacc",
                                    f"mode={mode},amount={amt},delta={prune_keep_acc_delta}",
                                    f"{det_before:.4f}",
                                    f"{det_after:.4f}",
                                    f"{ber:.4f}",
                                    f"{acc_before:.4f}",
                                    f"{acc_sel:.4f}",
                                    f"{(acc_before-acc_sel):.4f}",
                                    int(len(bits_arr)),
                                    int(nparams),
                                    f"{sparsity_sel:.6f}",
                                    "prune_keepacc",
                                ]
                            )
                            f.flush()

                    for amt in prune_amounts:
                        model_after = attack_prune(
                            state,
                            device_t,
                            amount=amt,
                            num_classes=num_classes,
                            model_name=model_name,
                            prune_mode=mode,
                            image_size=image_size,
                        )
                        det_after, ber = det_and_ber(model_after, positions, bits_arr)
                        acc_after = accuracy_on_loader(model_after, test_loader, device_t)
                        writer.writerow(
                            [
                                time.time(),
                                shard_key,
                                seed,
                                "prune",
                                f"mode={mode},amount={amt}",
                                f"{det_before:.4f}",
                                f"{det_after:.4f}",
                                f"{ber:.4f}",
                                f"{acc_before:.4f}",
                                f"{acc_after:.4f}",
                                f"{(acc_before-acc_after):.4f}",
                                int(len(bits_arr)),
                                int(nparams),
                                f"{measure_sparsity(model_after):.6f}",
                                "prune",
                            ]
                        )
                        f.flush()

                # noise
                for sigma in noise_sigmas:
                    model_after = attack_noise(
                        state,
                        device_t,
                        sigma=sigma,
                        seed=seed,
                        num_classes=num_classes,
                        model_name=model_name,
                        image_size=image_size,
                    )
                    det_after, ber = det_and_ber(model_after, positions, bits_arr)
                    acc_after = accuracy_on_loader(model_after, test_loader, device_t)
                    writer.writerow(
                        [
                            time.time(),
                            shard_key,
                            seed,
                            "noise",
                            f"sigma={sigma}",
                            f"{det_before:.4f}",
                            f"{det_after:.4f}",
                            f"{ber:.4f}",
                            f"{acc_before:.4f}",
                            f"{acc_after:.4f}",
                            f"{(acc_before-acc_after):.4f}",
                            int(len(bits_arr)),
                            int(nparams),
                            f"{measure_sparsity(model_after):.6f}",
                            "noise",
                        ]
                    )
                    f.flush()

                # quantize
                for nb in quant_bits:
                    model_after = attack_quantize(
                        state,
                        device_t,
                        n_bits=nb,
                        num_classes=num_classes,
                        model_name=model_name,
                        image_size=image_size,
                    )
                    det_after, ber = det_and_ber(model_after, positions, bits_arr)
                    acc_after = accuracy_on_loader(model_after, test_loader, device_t)
                    writer.writerow(
                        [
                            time.time(),
                            shard_key,
                            seed,
                            "quantize",
                            f"bits={nb}",
                            f"{det_before:.4f}",
                            f"{det_after:.4f}",
                            f"{ber:.4f}",
                            f"{acc_before:.4f}",
                            f"{acc_after:.4f}",
                            f"{(acc_before-acc_after):.4f}",
                            int(len(bits_arr)),
                            int(nparams),
                            f"{measure_sparsity(model_after):.6f}",
                            "quantize",
                        ]
                    )
                    f.flush()

    if baseline_rows:
        dets = np.asarray([r[0] for r in baseline_rows], dtype=np.float32)
        accs = np.asarray([r[1] for r in baseline_rows], dtype=np.float32)
        sparsities = np.asarray([r[2] for r in baseline_rows], dtype=np.float32)
        print(
            "[summary] baseline metrics: "
            f"det_mean={float(dets.mean()):.4f}, det_min={float(dets.min()):.4f}, "
            f"acc_mean={float(accs.mean()):.4f}, acc_min={float(accs.min()):.4f}, "
            f"sparsity_mean={float(sparsities.mean()):.4f}"
        )
    print(f"[done] finished. CSV written to: {outcsv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saved", type=str, required=True, help="path to saved .pth")
    p.add_argument("--outdir", type=str, required=True, help="output directory for csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="list of seeds")
    p.add_argument("--finetune-epochs", type=int, nargs="+", default=[1], help="finetune epoch choices")
    p.add_argument("--prune-amounts", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="prune amounts")
    p.add_argument("--prune-modes", type=str, nargs="+", default=["global_unstructured"], help="prune modes")
    p.add_argument("--prune-keep-acc-delta", type=float, default=None, help="enable keep-acc pruning search with delta")
    p.add_argument("--prune-keep-grid", type=float, nargs="+", default=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
    p.add_argument("--noise-sigmas", type=float, nargs="+", default=[1e-4, 1e-3], help="noise sigmas")
    p.add_argument("--quant-bits", type=int, nargs="+", default=[8], help="quantization bitwidths")
    p.add_argument("--dataset-root", type=str, default="./data")
    p.add_argument("--allow-unsafe-load", action="store_true", help="allow unsafe torch.load fallback")  # FIX
    args = p.parse_args()

    run_grid(
        args.saved,
        args.outdir,
        args.device,
        args.seeds,
        args.finetune_epochs,
        args.prune_amounts,
        args.noise_sigmas,
        args.quant_bits,
        prune_modes=args.prune_modes,
        prune_keep_acc_delta=args.prune_keep_acc_delta,
        prune_keep_grid=args.prune_keep_grid,
        dataset_root=args.dataset_root,
        allow_unsafe_load=args.allow_unsafe_load,  # FIX
    )
