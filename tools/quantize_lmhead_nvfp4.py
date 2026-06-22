#!/usr/bin/env python3
"""Quantize MiMo lm_head BF16 -> NVFP4 (W4A16) overlay load-dir. Self-contained
(inlines the validated nvfp4 quant from quantize_oproj_mxfp8.py). CPU only, one
tensor at a time. Original checkpoint never modified. Format matches
ModelOptNvFp4W4A16LinearMethod (validated on sm_121, rel-err 0.008).
"""
import argparse, glob, json, os, sys
import torch

NVFP4_BLOCK_SIZE = 16; FLOAT4_E2M1_MAX = 6.0; FP8_E4M3_MAX = 448.0
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _cast_to_fp4_codes(a):
    code = torch.zeros_like(a, dtype=torch.uint8)
    code = torch.where((a > 0.25) & (a < 0.75), torch.tensor(1, dtype=torch.uint8), code)
    code = torch.where((a >= 0.75) & (a <= 1.25), torch.tensor(2, dtype=torch.uint8), code)
    code = torch.where((a > 1.25) & (a < 1.75), torch.tensor(3, dtype=torch.uint8), code)
    code = torch.where((a >= 1.75) & (a <= 2.5), torch.tensor(4, dtype=torch.uint8), code)
    code = torch.where((a > 2.5) & (a < 3.5), torch.tensor(5, dtype=torch.uint8), code)
    code = torch.where((a >= 3.5) & (a <= 5.0), torch.tensor(6, dtype=torch.uint8), code)
    code = torch.where(a > 5.0, torch.tensor(7, dtype=torch.uint8), code)
    return code

def nvfp4_quantize_cpu(w_bf16):
    out, k = w_bf16.shape
    assert k % NVFP4_BLOCK_SIZE == 0
    nb = k // NVFP4_BLOCK_SIZE
    x = w_bf16.to(torch.float32)
    global_amax = x.abs().max().clamp(min=1e-12)
    weight_scale_2 = (global_amax / (FLOAT4_E2M1_MAX * FP8_E4M3_MAX)).to(torch.float32)
    global_scale = (1.0 / weight_scale_2).to(torch.float32)
    xb = x.view(out, nb, NVFP4_BLOCK_SIZE)
    vec_max = xb.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = torch.clamp(scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    scale_f32 = scale_fp8.to(torch.float32)
    denom = scale_f32 * (1.0 / global_scale)
    output_scale = torch.where(denom == 0, torch.zeros_like(denom), 1.0 / denom)
    scaled = torch.clamp(xb * output_scale, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).view(out, k)
    sign = (scaled < 0).to(torch.uint8) << 3
    codes = _cast_to_fp4_codes(scaled.abs())
    nibbles = (sign | codes).view(out, k)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale_fp8.view(out, nb), weight_scale_2.reshape(())

def nvfp4_dequant_cpu(packed, wscale_fp8, wscale2):
    out, half = packed.shape; k = half * 2; nb = wscale_fp8.shape[1]; bs = k // nb
    low = packed & 0x0F; high = (packed >> 4) & 0x0F
    inter = torch.stack((low, high), dim=2).view(out, k)
    sign = torch.where((inter & 0x08) > 0, -1.0, 1.0)
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
    vals = lut[(inter & 0x07).to(torch.long)] * sign
    bscale = wscale_fp8.to(torch.float32) * wscale2.to(torch.float32)
    return (vals.view(out, nb, bs) * bscale.unsqueeze(-1)).view(out, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.torch import safe_open, save_file
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.snap, "model.safetensors.index.json")))
    wm = idx["weight_map"]; assert "lm_head.weight" in wm
    lmshard = wm["lm_head.weight"]  # original shard that contains the bf16 lm_head.weight
    with safe_open(os.path.join(a.snap, lmshard), framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")
    print(f"[load] lm_head.weight {tuple(w.shape)} {w.dtype}")
    packed, wscale, wscale2 = nvfp4_quantize_cpu(w.to(torch.bfloat16))
    rel = ((nvfp4_dequant_cpu(packed, wscale, wscale2).float() - w.float()).norm() / w.float().norm()).item()
    print(f"[quant] weight {tuple(packed.shape)}u8 scale {tuple(wscale.shape)}{wscale.dtype} wscale2={wscale2.item():.3e} rel-err={rel:.4f}")
    nt = {"lm_head.weight": packed.contiguous(), "lm_head.weight_scale": wscale.contiguous(),
          # W4A16 create_weights makes weight_scale_2 with shape [len(output_partition_sizes)]=[1];
          # the embedding weight_loader asserts exact shape match -> store [1], not scalar [].
          "lm_head.weight_scale_2": wscale2.to(torch.float32).reshape(1).contiguous()}
    shard = "model-lmhead-nvfp4.safetensors"
    save_file(nt, os.path.join(a.out, shard), metadata={"format": "pt"})
    new_wm = {k: v for k, v in wm.items() if k != "lm_head.weight"}
    for k in nt: new_wm[k] = shard
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wm},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"))
    # symlink everything EXCEPT index/config/hf_quant AND the shard that held the
    # bf16 lm_head.weight (that shard we rewrite filtered, else the weights
    # iterator yields the bf16 lm_head.weight from it and clobbers the nvfp4 param).
    skip = {"model.safetensors.index.json", "config.json", "hf_quant_config.json", lmshard}
    for p in glob.glob(os.path.join(a.snap, "*")):
        b = os.path.basename(p); dst = os.path.join(a.out, b)
        if b not in skip and not os.path.exists(dst):
            os.symlink(os.path.realpath(p), dst)
    # filtered copy of lmshard: all tensors EXCEPT lm_head.weight
    keep = {}
    with safe_open(os.path.join(a.snap, lmshard), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "lm_head.weight":
                continue
            keep[k] = f.get_tensor(k)
    _dst = os.path.join(a.out, lmshard)
    # CRITICAL: remove any pre-existing path first. If it is a symlink to the
    # original blob, save_file would follow it and CORRUPT the source checkpoint.
    if os.path.lexists(_dst):
        os.remove(_dst)
    save_file(keep, _dst, metadata={"format": "pt"})
    print(f"[write] filtered {lmshard} ({len(keep)} tensors, lm_head.weight dropped)")
    cfg = json.load(open(os.path.join(a.snap, "config.json")))
    q = cfg.get("quantization_config", cfg.get("quantization"))
    if q is not None:
        q.setdefault("quantized_layers", {})["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        for key in ("ignore", "exclude_modules"):
            if isinstance(q.get(key), list):
                q[key] = [e for e in q[key] if e not in ("lm_head", "lm_head*")]
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    hf = os.path.join(a.snap, "hf_quant_config.json")
    if os.path.exists(hf):
        qc = json.load(open(hf)); qq = qc.get("quantization", qc)
        qq.setdefault("quantized_layers", {})["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        for key in ("ignore", "exclude_modules"):
            if isinstance(qq.get(key), list):
                qq[key] = [e for e in qq[key] if e not in ("lm_head", "lm_head*")]
        json.dump(qc, open(os.path.join(a.out, "hf_quant_config.json"), "w"), indent=1)
    print(f"[done] overlay at {a.out}  (lm_head nvfp4 W4A16, {len(new_wm)} index entries)")

if __name__ == "__main__":
    main()
