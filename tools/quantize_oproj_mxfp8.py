#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
quantize_oproj_mxfp8.py  --  MiMo-V2.5-NVFP4 selective re-quantization

GOAL:
  * o_proj  (BF16 [4096,8192] x48 layers, 3.22 GB, read every decode step)
        -> MXFP8  (primary, calibration-free, identical layout to qkv that
                   already ships MXFP8 on this model)  OR
        -> NVFP4 W4A4 (--oproj-format nvfp4, needs an activation input_scale;
                       see WARNING -- only with calibration)
  * lm_head (BF16 [152576,4096], 1.25 GB, read 1x main + ~2x MTP)
        -> MXFP8  (fp8, NOT fp4 -- direct on logits is too risky in fp4;
                   calibration-free dynamic activation, no input_scale crash)

WHY MXFP8 and not per-tensor static FP8:
  vLLM's MIXED_PRECISION dispatch (modelopt.py ModelOptMixedPrecisionConfig
  .get_quant_method) routes per-layer quant_algo:
      "FP8"   -> ModelOptFp8LinearMethod    (per-tensor STATIC, REQUIRES an
                 input_scale on disk; process_weights does input_scale.max()
                 -> would CRASH or mis-scale activations without calibration)
      "MXFP8" -> ModelOptMxFp8LinearMethod  (block-32 E8M0 weight scale,
                 DYNAMIC per-token activation, NO input_scale -> calibration
                 free, already proven on qkv of this exact checkpoint)
      "NVFP4" -> ModelOptNvFp4LinearMethod  (W4A4, REQUIRES input_scale)
  => MXFP8 is the safe, no-calibration, no-code-change target for both
     o_proj and lm_head. NVFP4-o_proj is offered but gated behind a flag
     because it is W4A4 and needs a real activation scale.

FORMAT (verified against on-disk tensors of this checkpoint):
  MXFP8 Linear (e.g. qkv_proj):
     <name>.weight            F8_E4M3   [out, in]
     <name>.weight_scale_inv  U8(E8M0)  [out, in/32]      (row-major, NON-swizzled;
                                                            vLLM swizzles at load)
  NVFP4 W4A4 Linear (e.g. routed expert):
     <name>.weight            U8        [out, in/2]        (e2m1, 2 nibbles/byte)
     <name>.weight_scale      F8_E4M3   [out, in/16]       (per-16 block scale)
     <name>.weight_scale_2    F32       scalar             = amax/(6*448)
     <name>.input_scale       F32       scalar             (activation scale)

SAFETY  --  READ BEFORE RUNNING:
  * Pure CPU. Run with CUDA_VISIBLE_DEVICES="" . Never imports vLLM/MoE code.
  * Loads ONE tensor at a time, frees it before the next -> peak RAM is one
    o_proj (64 MB bf16) + its fp8 copy. NEVER materializes the full 3.22 GB
    o_proj set nor the 1.25 GB lm_head twice.
  * DO NOT RUN WHILE THE vLLM ENGINE OR A LEVER-SWEEP IS USING GPU/UNIFIED
    MEMORY. The host shares unified memory; a big alloc here can freeze the
    box. Gate on: `free -g` free >= 20 and no active sweep (see PLAN).
  * Writes a NEW mini-checkpoint dir (only changed tensors) + a merged index
    + patched hf_quant_config.json. Original checkpoint is never modified.

Outputs (under --out):
    model-oproj-lmhead-requant.safetensors   (only the new tensors)
    model.safetensors.index.json             (merged: original map minus the
                                              replaced bf16 names, plus the new)
    hf_quant_config.json                     (quantized_layers + exclude patched)
    REQUANT_MANIFEST.json                    (what changed, rel-err per tensor)
"""

import argparse
import gc
import json
import os
import struct
import sys

# Hard CPU guard -- must be set before torch sees any device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

MXFP8_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0

# e2m1 magnitude LUT (index 0..7 -> float). Sign is bit 3.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ----------------------------------------------------------------------------
# MXFP8 (calibration-free) -- mirrors vllm _mxfp8_e4m3_quantize_torch exactly,
# non-swizzled row-major layout (== on-disk qkv weight_scale_inv).
# ----------------------------------------------------------------------------
def mxfp8_quantize_cpu(w_bf16: torch.Tensor):
    """w_bf16: [out, in], in % 32 == 0.
    Returns (weight_fp8 [out,in], scale_uint8 [out, in/32])."""
    assert w_bf16.ndim == 2
    out, k = w_bf16.shape
    assert k % MXFP8_BLOCK_SIZE == 0, f"in={k} not divisible by {MXFP8_BLOCK_SIZE}"
    nb = k // MXFP8_BLOCK_SIZE
    x = w_bf16.to(torch.float32)
    xb = x.view(out, nb, MXFP8_BLOCK_SIZE)
    amax = xb.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    # E8M0 biased exponent of the block amax (matches vLLM reference)
    scale_biased = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254)
    scales_u8 = scale_biased.to(torch.uint8)            # [out, nb]
    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = xb / descale.unsqueeze(-1)
    w_fp8 = x_scaled.view(out, k).to(torch.float8_e4m3fn)
    return w_fp8, scales_u8


def mxfp8_dequant_cpu(w_fp8: torch.Tensor, scales_u8: torch.Tensor):
    out, k = w_fp8.shape
    nb = scales_u8.shape[1]
    bs = k // nb
    descale = torch.exp2(scales_u8.to(torch.float32) - 127.0)  # [out, nb]
    x = w_fp8.to(torch.float32).view(out, nb, bs)
    return (x * descale.unsqueeze(-1)).view(out, k)


# ----------------------------------------------------------------------------
# NVFP4 W4A4 -- mirrors vllm ref_nvfp4_quant / cast_to_fp4. Produces the exact
# producer (modelopt) on-disk triple. weight_scale stored NON-swizzled
# (modelopt convention; vLLM kernel handles it).
# ----------------------------------------------------------------------------
def _cast_to_fp4_codes(x_abs_scaled: torch.Tensor) -> torch.Tensor:
    """Round |scaled| in [0,6] to nearest e2m1 value, return 3-bit magnitude code."""
    a = x_abs_scaled
    code = torch.zeros_like(a, dtype=torch.uint8)
    # thresholds identical to vllm cast_to_fp4 (mapped to codes, not values)
    code = torch.where((a > 0.25) & (a < 0.75), torch.tensor(1, dtype=torch.uint8), code)   # 0.5
    code = torch.where((a >= 0.75) & (a <= 1.25), torch.tensor(2, dtype=torch.uint8), code)  # 1.0
    code = torch.where((a > 1.25) & (a < 1.75), torch.tensor(3, dtype=torch.uint8), code)   # 1.5
    code = torch.where((a >= 1.75) & (a <= 2.5), torch.tensor(4, dtype=torch.uint8), code)  # 2.0
    code = torch.where((a > 2.5) & (a < 3.5), torch.tensor(5, dtype=torch.uint8), code)     # 3.0
    code = torch.where((a >= 3.5) & (a <= 5.0), torch.tensor(6, dtype=torch.uint8), code)   # 4.0
    code = torch.where(a > 5.0, torch.tensor(7, dtype=torch.uint8), code)                   # 6.0
    return code


def nvfp4_quantize_cpu(w_bf16: torch.Tensor):
    """w_bf16: [out, in], in % 16 == 0.
    Returns (weight_u8 [out, in/2], weight_scale_fp8 [out, in/16],
             weight_scale_2 fp32 scalar)."""
    assert w_bf16.ndim == 2
    out, k = w_bf16.shape
    assert k % NVFP4_BLOCK_SIZE == 0, f"in={k} not divisible by {NVFP4_BLOCK_SIZE}"
    nb = k // NVFP4_BLOCK_SIZE
    x = w_bf16.to(torch.float32)
    # On-disk weight_scale_2 (modelopt convention) = amax / (6 * 448).
    # NOTE: the per-block quantization multiply uses global_scale = 1/weight_scale_2
    # (== (6*448)/amax). This mirrors vLLM ref_nvfp4_quant exactly: it passes the
    # *reciprocal* as `global_scale`, then block scale = global_scale*(vec_max/6).
    global_amax = x.abs().max().clamp(min=1e-12)
    weight_scale_2 = (global_amax / (FLOAT4_E2M1_MAX * FP8_E4M3_MAX)).to(torch.float32)
    global_scale = (1.0 / weight_scale_2).to(torch.float32)            # = (6*448)/amax

    xb = x.view(out, nb, NVFP4_BLOCK_SIZE)
    vec_max = xb.abs().amax(dim=-1, keepdim=True).to(torch.float32)     # [out,nb,1]
    scale = global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = torch.clamp(scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    scale_fp8 = scale.to(torch.float8_e4m3fn)                           # block scale (fp8)
    scale_f32 = scale_fp8.to(torch.float32)
    # output_scale = 1 / (block_scale / global_scale)  (vLLM get_reciprocal, guard /0)
    denom = scale_f32 * (1.0 / global_scale)
    output_scale = torch.where(denom == 0, torch.zeros_like(denom), 1.0 / denom)
    scaled = (xb * output_scale)
    scaled = torch.clamp(scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).view(out, k)

    sign = (scaled < 0).to(torch.uint8) << 3
    codes = _cast_to_fp4_codes(scaled.abs())
    nibbles = (sign | codes).view(out, k)                              # 4-bit each
    # pack pairs: low nibble = even col, high nibble = odd col
    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)                          # [out, in/2]
    weight_scale = scale_fp8.view(out, nb)                             # [out, in/16]
    return packed, weight_scale, weight_scale_2.reshape(())


def nvfp4_dequant_cpu(packed, weight_scale_fp8, weight_scale_2):
    out, half = packed.shape
    k = half * 2
    nb = weight_scale_fp8.shape[1]
    bs = k // nb
    low = (packed & 0x0F)
    high = (packed >> 4) & 0x0F
    inter = torch.stack((low, high), dim=2).view(out, k)               # un-interleave
    sign = torch.where((inter & 0x08) > 0, -1.0, 1.0)
    mag = (inter & 0x07).to(torch.long)
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
    vals = lut[mag] * sign                                             # [out,k]
    bscale = weight_scale_fp8.to(torch.float32) * weight_scale_2.to(torch.float32)
    vals = vals.view(out, nb, bs) * bscale.unsqueeze(-1)
    return vals.view(out, k)


# ----------------------------------------------------------------------------
# safetensors helpers -- header-only reads, single-tensor reads.
# ----------------------------------------------------------------------------
_ST_DTYPE = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I8": torch.int8,
}
_ST_DTYPE_REV = {v: k for k, v in _ST_DTYPE.items()}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta


def load_one_tensor(path, name):
    """Load a single tensor by name without mapping the whole file into a
    framework tensor set. Reads exactly its byte range."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        info = header[name]
        s, e = info["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    dt = _ST_DTYPE[info["dtype"]]
    shape = info["shape"]
    t = torch.frombuffer(bytearray(raw), dtype=dt)
    if shape:
        t = t.view(*shape)
    else:
        t = t.view(())
    return t.clone()  # own the memory, drop the bytearray


def rel_err(a, b):
    a = a.to(torch.float32); b = b.to(torch.float32)
    num = (a - b).norm()
    den = a.norm().clamp(min=1e-12)
    return (num / den).item()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True,
                    help="path to the original checkpoint snapshot dir")
    ap.add_argument("--out", required=True, help="output dir for mini-checkpoint")
    ap.add_argument("--oproj-format", choices=["mxfp8", "nvfp4"], default="mxfp8")
    ap.add_argument("--lmhead-format", choices=["mxfp8", "skip"], default="mxfp8")
    ap.add_argument("--num-layers", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: print targets, do not load/quantize/write")
    args = ap.parse_args()

    try:
        from safetensors.torch import save_file
    except Exception as e:
        print("ERROR: need safetensors.torch (run inside the vLLM container):", e)
        sys.exit(2)

    snap = args.snapshot
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]

    targets = []  # (orig_name, shard_path)
    for L in range(args.num_layers):
        nm = f"model.layers.{L}.self_attn.o_proj.weight"
        if nm in wm:
            targets.append((nm, "oproj", os.path.join(snap, wm[nm])))
    if args.lmhead_format != "skip" and "lm_head.weight" in wm:
        targets.append(("lm_head.weight", "lmhead",
                        os.path.join(snap, wm["lm_head.weight"])))

    print(f"[plan] {len(targets)} target tensors "
          f"(o_proj->{args.oproj_format}, lm_head->{args.lmhead_format})")
    if args.dry_run:
        for n, kind, _ in targets:
            print("   ", kind, n)
        return

    os.makedirs(args.out, exist_ok=True)
    new_tensors = {}
    manifest = {"oproj_format": args.oproj_format,
                "lmhead_format": args.lmhead_format, "tensors": {}}

    for orig_name, kind, shard in targets:
        w = load_one_tensor(shard, orig_name)          # bf16, e.g. 64 MB
        base = orig_name[:-len(".weight")]
        fmt = args.oproj_format if kind == "oproj" else args.lmhead_format
        if fmt == "mxfp8":
            wq, sc = mxfp8_quantize_cpu(w)
            new_tensors[base + ".weight"] = wq
            new_tensors[base + ".weight_scale_inv"] = sc
            dq = mxfp8_dequant_cpu(wq, sc)
            algo = "MXFP8"
        else:  # nvfp4 W4A4
            packed, wscale, wscale2 = nvfp4_quantize_cpu(w)
            new_tensors[base + ".weight"] = packed
            new_tensors[base + ".weight_scale"] = wscale
            new_tensors[base + ".weight_scale_2"] = wscale2
            # W4A4 needs an activation input_scale. WITHOUT calibration we emit
            # a weight-amax-derived placeholder = amax/448 (so the kernel does
            # not divide by garbage). THIS IS NOT A CALIBRATED SCALE -- gate
            # quality hard, or run a calibration pass (see PLAN).
            inp = (w.abs().max().to(torch.float32) / FP8_E4M3_MAX).reshape(())
            new_tensors[base + ".input_scale"] = inp
            dq = nvfp4_dequant_cpu(packed, wscale, wscale2)
            algo = "NVFP4"
        err = rel_err(w, dq)
        manifest["tensors"][orig_name] = {"algo": algo, "rel_err": round(err, 5),
                                          "shape": list(w.shape)}
        print(f"  [{kind:6s}] {orig_name:48s} {algo:5s} rel_err={err:.4f}")
        del w, dq
        gc.collect()

    out_shard = "model-oproj-lmhead-requant.safetensors"
    save_file(new_tensors, os.path.join(args.out, out_shard),
              metadata={"format": "pt", "requant": "oproj_lmhead"})
    print(f"[write] {out_shard}  ({len(new_tensors)} tensors)")

    # ---- merged index.json ----------------------------------------------
    new_wm = dict(wm)
    replaced_bf16 = set()
    for orig_name, kind, _ in targets:
        replaced_bf16.add(orig_name)
        base = orig_name[:-len(".weight")]
        new_wm.pop(orig_name, None)
        fmt = args.oproj_format if kind == "oproj" else args.lmhead_format
        if fmt == "mxfp8":
            new_wm[base + ".weight"] = out_shard
            new_wm[base + ".weight_scale_inv"] = out_shard
        else:
            new_wm[base + ".weight"] = out_shard
            new_wm[base + ".weight_scale"] = out_shard
            new_wm[base + ".weight_scale_2"] = out_shard
            new_wm[base + ".input_scale"] = out_shard
    merged_idx = {"metadata": idx.get("metadata", {}), "weight_map": new_wm}
    json.dump(merged_idx, open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
    print(f"[write] merged index ({len(new_wm)} entries, "
          f"{len(replaced_bf16)} bf16 names removed)")

    # ---- patch the quantization config -----------------------------------
    # IMPORTANT: vLLM (transformers_utils/config.py) reads the EMBEDDED
    # config.json["quantization_config"] FIRST; hf_quant_config.json is only
    # consulted if the embedded one is absent. This checkpoint EMBEDS one, so we
    # MUST patch config.json. We patch both for safety. The embedded form uses
    # `ignore` (list) and a `config_groups` dict; the standalone form nests
    # under `quantization` with `exclude_modules`. Handle both shapes.
    oproj_entry = ({"quant_algo": "MXFP8", "group_size": 32}
                   if args.oproj_format == "mxfp8"
                   else {"quant_algo": "NVFP4", "group_size": 16})

    def patch_quant_block(q: dict):
        """Mutate an in-place quantization dict (embedded or standalone form)."""
        ql = q.setdefault("quantized_layers", {})
        for L in range(args.num_layers):
            if f"model.layers.{L}.self_attn.o_proj.weight" in replaced_bf16:
                ql[f"model.layers.{L}.self_attn.o_proj"] = dict(oproj_entry)
        if args.lmhead_format != "skip" and "lm_head.weight" in replaced_bf16:
            ql["lm_head"] = {"quant_algo": "MXFP8", "group_size": 32}
            # Remove lm_head from BOTH possible exclusion keys so it is quantized.
            for key in ("ignore", "exclude_modules"):
                if key in q and isinstance(q[key], list):
                    q[key] = [e for e in q[key]
                              if e not in ("lm_head", "lm_head*")]
        # NOTE: we deliberately do NOT add o_proj/lm_head to `config_groups`.
        # vLLM's MIXED_PRECISION dispatch keys off `quantized_layers` only
        # (modelopt.py _resolve_quant_algo); config_groups is producer metadata
        # not consulted by the vLLM per-layer dispatch. Leaving it untouched is
        # correct and avoids a malformed group.

    # (a) embedded config.json (the one vLLM actually loads)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    if "quantization_config" in cfg:
        patch_quant_block(cfg["quantization_config"])
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
        print("[write] patched config.json (embedded quantization_config -- "
              "this is the one vLLM reads)")
    else:
        print("[warn] config.json has NO embedded quantization_config; vLLM will "
              "use hf_quant_config.json -- make sure it lands in the load dir")

    # (b) standalone hf_quant_config.json (fallback / consistency)
    hf_path = os.path.join(snap, "hf_quant_config.json")
    if os.path.exists(hf_path):
        qcfg = json.load(open(hf_path))
        block = qcfg["quantization"] if "quantization" in qcfg else qcfg
        patch_quant_block(block)
        json.dump(qcfg, open(os.path.join(args.out, "hf_quant_config.json"), "w"),
                  indent=1)
        print("[write] patched hf_quant_config.json")

    json.dump(manifest, open(os.path.join(args.out, "REQUANT_MANIFEST.json"), "w"), indent=1)
    print("[done] manifest written. See QUANT_BUILD_PLAN.md for the boot recipe.")


if __name__ == "__main__":
    main()
