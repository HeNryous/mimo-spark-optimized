# Tools

## `quantize_oproj_mxfp8.py` — selective o_proj -> MXFP8 re-quantization

MiMo-V2.5-NVFP4 ships its routed experts in NVFP4 but keeps `o_proj` (and
`lm_head`) in BF16. Re-quantizing the per-layer `o_proj` weights to **MXFP8**
(block-32 E8M0 scale, calibration-free dynamic activation — the same format the
checkpoint already uses for `qkv_proj`) frees ~1.6 GB and yields **+8.5% KV-cache
capacity** in measurement, with quality held (coherent; reliability-bench neutral).

It does **not** speed up decode: single-stream decode on this 2x GB10 / TP=2 setup
is latency/comm-bound, not weight-bandwidth-bound, so halving o_proj bytes gives
~0% tok/s. The win is purely KV capacity.

### What it does

- Pure-CPU, one tensor at a time (peak RAM ~ one o_proj + its fp8 copy). Never
  imports vLLM/MoE code, never materializes the full weight set.
- Writes a **new mini-checkpoint** (only the changed tensors) + a merged
  `model.safetensors.index.json` + patched `hf_quant_config.json`. The original
  checkpoint is never modified — you load it as an overlay on top of the base.

### Usage

```bash
# Run on CPU only; do NOT run while the vLLM engine is using GPU/unified memory
# (shared unified memory — a large alloc can freeze the box). Require free >= 20G.
CUDA_VISIBLE_DEVICES="" python3 quantize_oproj_mxfp8.py \
    --in   <PATH_TO_MIMO_NVFP4_CHECKPOINT> \
    --out  <PATH_TO_OVERLAY_OUTPUT_DIR>
```

Then point your loader at the overlay dir (containing only the re-quantized
tensors + merged index) layered over the original snapshot, on **both** nodes.

### lm_head (EXPERIMENTAL — not drop-in)

The script can also target `lm_head` (`--lm-head`), but **vLLM's `ParallelLMHead`
/ logits path does not accept a quantized weight** — it has no `weight_scale_inv`
parameter and boot fails with:

```
ValueError: no parameter 'lm_head.weight_scale_inv' in ... ParallelLMHead has only {lm_head.weight}
```

Enabling lm_head quant therefore requires a separate vLLM code patch that extends
`ParallelLMHead` to register and consume the MXFP8 scale. That patch is **not**
included here. Treat the lm_head path as experimental / for those willing to
patch vLLM. The `o_proj`-only path is the validated one.
