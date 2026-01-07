#!/usr/bin/env python3
"""
merge_checkpoints.py

从一个 PoLO 保存文件中读取字段并写为一个新的导出文件。
增强：
- --export public|proof|both
  public: 只输出 final_state + minimal meta（不含 positions/bits/key/target_map/chain）
  proof:  保留 checkpoints 等证明材料
- 写入 provenance
"""

import argparse, os, time, torch, json
from utils import safe_torch_load, get_run_env, write_json

PUBLIC_KEYS = ["final_state", "meta", "created_at", "run_id", "config"]
PROOF_KEYS = ["checkpoints", "meta", "final_state", "created_at", "notes", "config", "mu", "id_P", "run_id"]

SENSITIVE_IN_CKPT = {"positions","bits","key","target_map","chain_H","prev_proj_hash","positions_hash","embed_log"}

def strip_checkpoints_sensitive(ckpts: dict) -> dict:
    if not isinstance(ckpts, dict):
        return ckpts
    out = {}
    for k, v in ckpts.items():
        if not isinstance(v, dict):
            out[k] = v
            continue
        out[k] = {kk: vv for kk, vv in v.items() if kk not in SENSITIVE_IN_CKPT}
    return out

def write_export(outpath: str, data: dict, saved_from: str, export: str):
    top = {}
    keys = PUBLIC_KEYS if export == "public" else PROOF_KEYS
    for k in keys:
        if k in data:
            top[k] = data[k]

    # provenance
    top["_merged_from"] = os.path.abspath(saved_from)
    top["_merged_by"] = "merge_checkpoints.py"
    top["_merged_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    top["_env"] = get_run_env()

    # if public: strip checkpoints or remove
    if export == "public":
        if "checkpoints" in top:
            top["checkpoints"] = strip_checkpoints_sensitive(top["checkpoints"])

    torch.save(top, outpath)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saved", type=str, required=True, help="input saved .pth file")
    p.add_argument("--out", type=str, required=True, help="output base path (dir or file)")
    p.add_argument("--export", type=str, default="proof", choices=["public","proof","both"])
    p.add_argument("--allow-unsafe-load", action="store_true", help="allow unsafe torch.load fallback")
    args = p.parse_args()

    data = safe_torch_load(args.saved, allow_unsafe=args.allow_unsafe_load)

    if args.export == "both":
        out_public = args.out
        out_proof = args.out
        if out_public.endswith(".pth"):
            out_public = out_public.replace(".pth", "_public.pth")
            out_proof = out_proof.replace(".pth", "_proof.pth")
        else:
            os.makedirs(args.out, exist_ok=True)
            out_public = os.path.join(args.out, "export_public.pth")
            out_proof = os.path.join(args.out, "export_proof.pth")

        write_export(out_public, data, args.saved, export="public")
        write_export(out_proof, data, args.saved, export="proof")
        print(f"[info] wrote public: {out_public}")
        print(f"[info] wrote proof:   {out_proof}")
    else:
        outpath = args.out
        if not outpath.endswith(".pth"):
            os.makedirs(outpath, exist_ok=True)
            outpath = os.path.join(outpath, f"export_{args.export}.pth")
        write_export(outpath, data, args.saved, export=args.export)
        print(f"[info] wrote {args.export}: {outpath}")

if __name__ == "__main__":
    main()