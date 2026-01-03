#!/usr/bin/env python3
"""
utils.py - 公共工具（安全加载、哈希/KDF、投影、(反)序列化、归一化、自适应DP、运行元信息）

本次修正重点：
- safe_torch_load 兼容 PyTorch 2.6+ 的 weights_only 默认行为
- 正确使用 torch.serialization.safe_globals（context manager）
- 仅当 allow_unsafe=True 时才回退到 weights_only=False（有代码执行风险）
"""

import os
import sys
import json
import time
import math
import uuid
import warnings
import hashlib
import random
from typing import Any, Dict, Iterable, Tuple, List, Optional

import numpy as np
import torch


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_run_env() -> Dict[str, Any]:
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "python": sys.version.replace("\n", " "),
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device_name_0"] = None
    return info


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:8]}"


# -------------------------
# Safe torch load (FIXED)
# -------------------------
def _torch_load_compat(path: str, map_location="cpu", weights_only: Optional[bool] = None) -> Any:
    """
    Call torch.load with best-effort compatibility across torch versions.

    - torch<2.0: no weights_only kwarg
    - torch>=2.0: may accept weights_only
    """
    if weights_only is None:
        return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # older torch doesn't accept weights_only
        return torch.load(path, map_location=map_location)


def safe_torch_load(path: str, allow_unsafe: bool = False, map_location: str = "cpu") -> Any:
    """
    Safe-ish loader:
    1) Try normal torch.load (older torch) / or weights_only=True (newer torch)
    2) If fails in torch>=2.6 due to "Unsupported global", retry with safe_globals allowlist
    3) Only if allow_unsafe=True, fall back to weights_only=False (RCE risk)
    """
    first_err = None

    # Attempt 1: prefer weights_only=True when available
    if hasattr(torch.serialization, "add_safe_globals"):
        torch_version_mod = getattr(torch, "torch_version", None)
        torch_version_cls = getattr(torch_version_mod, "TorchVersion", None) if torch_version_mod else None
        if torch_version_cls is not None:
            torch.serialization.add_safe_globals([torch_version_cls])
    try:
        return _torch_load_compat(path, map_location=map_location, weights_only=True)
    except Exception as e:
        first_err = e
        warnings.warn(f"[warn] torch.load(weights_only=True) failed: {e}")

    # Attempt 2: safe_globals allowlist (torch>=2.6)
    safe_globals_ctx = getattr(torch.serialization, "safe_globals", None)
    if safe_globals_ctx is not None:
        allowlist = []
        try:
            import numpy  # noqa: F401
            allowlist.append(np.core.multiarray._reconstruct)  # type: ignore[attr-defined]
        except Exception:
            pass

        torch_version_mod = getattr(torch, "torch_version", None)
        torch_version_cls = getattr(torch_version_mod, "TorchVersion", None) if torch_version_mod else None
        if torch_version_cls is not None:
            allowlist.append(torch_version_cls)

        try:
            with safe_globals_ctx(allowlist):
                return _torch_load_compat(path, map_location=map_location, weights_only=True)
        except Exception as e2:
            warnings.warn(f"[warn] torch.load(weights_only=True) with safe_globals failed: {e2}")

    # Attempt 3: last resort - unsafe load
    if not allow_unsafe:
        raise RuntimeError(
            "safe_torch_load failed. Set --allow-unsafe-load only if you trust the file.\n"
            f"First error: {first_err}"
        )

    warnings.warn(
        "[warn] Falling back to torch.load(..., weights_only=False). "
        "This may execute arbitrary code if the file is malicious."
    )
    return _torch_load_compat(path, map_location=map_location, weights_only=False)

# (其余内容保持不变)



# -------------------------
# Chain hash / KDF
# -------------------------
def derive_chain_hash(W_prev_proj_bytes: bytes, x: int, mu: Any, id_P: Any) -> str:
    """
    chain_H = H( prev_projection || shard_index || mu || id_P )
    """
    m = hashlib.sha256()
    if isinstance(W_prev_proj_bytes, (bytes, bytearray)):
        m.update(W_prev_proj_bytes)
    else:
        try:
            m.update(bytes(W_prev_proj_bytes))
        except Exception:
            m.update(str(W_prev_proj_bytes).encode())

    m.update(int(x).to_bytes(4, "little", signed=False))
    m.update(str(mu).encode() if mu is not None else b"")
    m.update(str(id_P).encode() if id_P is not None else b"")
    return m.hexdigest()


def kdf_from_hash(hexhash: str, n_bits: int = 1024, key_len: int = 32) -> Tuple[np.ndarray, bytes]:
    """
    Deterministic stream from repeated sha256. Output:
    - bits: np.uint8 array of length n_bits
    - key_bytes: length key_len
    """
    out = b""
    state = hexhash.encode()
    need = (n_bits + 7) // 8 + key_len
    while len(out) < need:
        state = hashlib.sha256(state).digest()
        out += state
    wm_bytes = out[: (n_bits + 7) // 8]
    key_bytes = out[(n_bits + 7) // 8 : (n_bits + 7) // 8 + key_len]

    bits = np.unpackbits(np.frombuffer(wm_bytes, dtype=np.uint8))
    bits = bits[:n_bits].astype(np.uint8)
    return bits, key_bytes




# -------------------------
# Optional simple ECC (repetition code)
# -------------------------
def ecc_encode_repetition(payload_bits: np.ndarray, repeat: int, out_len: int) -> np.ndarray:
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    if repeat <= 1:
        out = payload_bits.copy()
    else:
        out = np.repeat(payload_bits, repeat)
    if len(out) >= out_len:
        return out[:out_len].astype(np.uint8)
    pad = np.zeros(out_len - len(out), dtype=np.uint8)
    return np.concatenate([out, pad]).astype(np.uint8)


def ecc_decode_repetition(code_bits: np.ndarray, repeat: int, payload_len: int) -> np.ndarray:
    code_bits = np.asarray(code_bits, dtype=np.uint8).reshape(-1)
    if repeat <= 1:
        return code_bits[:payload_len].astype(np.uint8)

    need = payload_len * repeat
    cb = code_bits[:need]
    if len(cb) < need:
        cb = np.concatenate([cb, np.zeros(need - len(cb), dtype=np.uint8)])
    cb = cb.reshape(payload_len, repeat)
    votes = (cb.sum(axis=1) >= (repeat / 2.0)).astype(np.uint8)
    return votes


# -------------------------
# Projection helpers
# -------------------------
def weights_projection_from_state_dict(state_dict: Dict, max_bytes: int = 8192) -> bytes:
    parts: List[bytes] = []
    for name in state_dict.keys():
        tensor = state_dict[name]
        try:
            arr = tensor.detach().cpu().numpy()
            parts.append(np.float32(arr.mean()).tobytes())
            parts.append(np.float32(arr.std()).tobytes())
        except Exception:
            parts.append(hashlib.sha256(str(name).encode()).digest()[:8])
            continue
    b = b"".join(parts)
    return b[:max_bytes] if len(b) > max_bytes else b


def weights_projection(model: torch.nn.Module, max_bytes: int = 8192) -> bytes:
    parts: List[bytes] = []
    for name, p in model.named_parameters():
        try:
            arr = p.detach().cpu().numpy()
            parts.append(np.float32(arr.mean()).tobytes())
            parts.append(np.float32(arr.std()).tobytes())
        except Exception:
            parts.append(hashlib.sha256(str(name).encode()).digest()[:8])
            continue
    b = b"".join(parts)
    return b[:max_bytes] if len(b) > max_bytes else b


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -------------------------
# Serialization helpers
# -------------------------
def serialize_checkpoints_for_saving(checkpoints: Dict) -> Dict:
    out = {}
    for k, v in checkpoints.items():
        pos_serial = [(name, idx.tolist()) for name, idx in v.get("positions", [])]
        bits = v.get("bits", [])
        bits_serial = bits.tolist() if hasattr(bits, "tolist") else list(bits)

        tm = v.get("target_map", None)
        tm_serial = {}
        if tm:
            for name_t, val in tm.items():
                if isinstance(val, tuple) and len(val) == 2:
                    signs, margin = val
                    signs_list = signs.tolist() if hasattr(signs, "tolist") else list(signs)
                    tm_serial[name_t] = {"signs": signs_list, "margin": float(margin)}
                else:
                    arr = np.asarray(val, dtype=np.float32)
                    tm_serial[name_t] = {"arr": arr.tolist()}

        out[k] = {
            "state": v.get("state"),
            "positions": pos_serial,
            "bits": bits_serial,
            "key": v.get("key", None),
            "target_map": tm_serial,
            "chain_H": v.get("chain_H", None),
            "prev_proj_hash": v.get("prev_proj_hash", None),
            "positions_hash": v.get("positions_hash", None),
            "test_acc": float(v.get("test_acc", -1.0)),
            "notes": v.get("notes", ""),
            "embed_log": v.get("embed_log", None),
            "bits_source": v.get("bits_source", None),
        }
    return out


def deserialize_checkpoints_loaded(ckpts_serial: Dict) -> Dict:
    out = {}
    for k, v in ckpts_serial.items():
        positions = []
        for name, idx_list in v.get("positions", []):
            positions.append((str(name), np.asarray(idx_list, dtype=np.int64)))
        bits = np.asarray(v.get("bits", []), dtype=np.uint8)

        tm_serial = v.get("target_map", None)
        target_map = {}
        if tm_serial:
            for name_t, info in tm_serial.items():
                if isinstance(info, dict) and "signs" in info:
                    signs = np.asarray(info["signs"], dtype=np.int8)
                    margin = float(info.get("margin", 0.05))
                    target_map[name_t] = (signs, margin)
                elif isinstance(info, dict) and "arr" in info:
                    arr = np.asarray(info.get("arr", []), dtype=np.float32)
                    target_map[name_t] = arr

        out[k] = {
            "state": v.get("state"),
            "positions": positions,
            "bits": bits,
            "key": v.get("key", None),
            "target_map": target_map,
            "chain_H": v.get("chain_H", None),
            "prev_proj_hash": v.get("prev_proj_hash", None),
            "positions_hash": v.get("positions_hash", None),
            "test_acc": v.get("test_acc", None),
            "notes": v.get("notes", ""),
            "embed_log": v.get("embed_log", None),
            "bits_source": v.get("bits_source", None),
        }
    return out


# -------------------------
# Normalizers
# -------------------------
def normalize_target_map(raw_target_map):
    if raw_target_map is None:
        return None
    norm = {}
    for name, v in raw_target_map.items():
        if isinstance(v, tuple) and len(v) == 2:
            signs = np.asarray(v[0]).astype(np.int8)
            margin = float(v[1])
            norm[name] = (signs, margin)
            continue
        if isinstance(v, dict):
            if "signs" in v and "margin" in v:
                norm[name] = (np.asarray(v["signs"], dtype=np.int8), float(v["margin"]))
                continue
            if "arr" in v:
                norm[name] = np.asarray(v["arr"], dtype=np.float32)
                continue
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v)
            if arr.dtype.kind in ("i", "u", "b"):
                norm[name] = (arr.astype(np.int8), 0.05)
            else:
                norm[name] = arr.astype(np.float32)
            continue
        if isinstance(v, str):
            import ast
            parsed = ast.literal_eval(v)
            arr = np.asarray(parsed)
            if arr.dtype.kind in ("i", "u", "b"):
                norm[name] = (arr.astype(np.int8), 0.05)
            else:
                norm[name] = arr.astype(np.float32)
            continue
        raise ValueError(f"Cannot normalize target_map entry for '{name}'; type={type(v)}")
    return norm


def normalize_positions(raw_positions):
    out = []
    if raw_positions is None:
        return out
    for item in raw_positions:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"positions item malformed: {item!r}")
        name, idxs = item
        out.append((str(name), np.asarray(idxs, dtype=np.int64)))
    return out


def positions_to_bytes(positions: List[Tuple[str, np.ndarray]]) -> bytes:
    chunks = []
    for name, idx in positions:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        chunks.append(name.encode("utf-8"))
        chunks.append(b":")
        chunks.append(idx.tobytes())
        chunks.append(b";")
    return b"".join(chunks)


def positions_hash_hex(positions: List[Tuple[str, np.ndarray]]) -> str:
    return hashlib.sha256(positions_to_bytes(positions)).hexdigest()


# -------------------------
# Adaptive DP helpers（续练用）
# -------------------------
def compute_param_importance(
    model: torch.nn.Module, dataloader, criterion, device, warmup_steps=50
) -> Dict[str, np.ndarray]:
    model.eval()
    param_names = [name for name, _ in model.named_parameters()]
    accum = {name: None for name in param_names}
    counts = 0
    device_t = device
    for i, (bx, by) in enumerate(dataloader):
        bx = bx.to(device_t)
        by = by.to(device_t)
        model.zero_grad(set_to_none=True)
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().cpu().abs().view(-1).numpy()
            if accum[name] is None:
                accum[name] = g
            else:
                accum[name] += g
        counts += 1
        if counts >= warmup_steps:
            break

    for name in list(accum.keys()):
        if accum[name] is None:
            accum[name] = np.zeros(1, dtype=np.float32)
        else:
            accum[name] = (accum[name] / max(1, counts)).astype(np.float32)

    all_vals = np.concatenate([v.flatten() for v in accum.values()]) if len(accum) > 0 else np.array([1.0])
    global_max = float(np.percentile(np.abs(all_vals), 99)) if all_vals.size > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
    normed = {name: (v / global_max).astype(np.float32) for name, v in accum.items()}
    return normed


def apply_adaptive_dp_noise(
    model: torch.nn.Module,
    positions: Iterable[Tuple[str, np.ndarray]],
    importance_map: Dict[str, np.ndarray],
    base_sigma: float = 1e-4,
    wm_ratio: float = 0.1,
    importance_scale: float = 2.0,
):
    wm_param_names = set([name for name, _ in positions])
    with torch.no_grad():
        for name, p in model.named_parameters():
            flat_n = p.view(-1).numel()
            if name in importance_map:
                imp = importance_map[name]
                imp_vec = imp if imp.size == flat_n else np.resize(imp, flat_n)
            else:
                imp_vec = np.zeros(flat_n, dtype=np.float32)

            if name in wm_param_names:
                sigma = base_sigma * float(wm_ratio)
                p.add_(torch.randn_like(p) * sigma)
            else:
                per_sigma = base_sigma * (1.0 + importance_scale * torch.from_numpy(imp_vec).to(p.device).float()).view(
                    p.shape
                )
                p.add_(torch.randn_like(p) * per_sigma)
    return model


def write_json(path: str, obj: Any):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
