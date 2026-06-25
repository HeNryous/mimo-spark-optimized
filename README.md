# MiMo-V2.5-NVFP4 on 2x NVIDIA DGX Spark — optimized vLLM build

> **Note on the prebuilt image:** `ghcr.io/henryous/mimo-spark-optimized:v1.0` (and `:latest`)
> are pinned to the v1.0 content (TK decode kernel). The **NS96 tuning** and **async-MTP unlock**
> below are in the source only — `docker pull` does *not* rebuild. **Build from this repo**
> (`docker build`) to get the latest optimizations baked in.


Running **MiMo-V2.5** (Xiaomi, ~310B-parameter Mixture-of-Experts, ~15B active,
NVFP4 weights) on **two NVIDIA DGX Spark** (GB10, `sm_121`, ARM aarch64) with
tensor parallelism (TP=2) over RoCE — and a set of custom kernels and config
levers that roughly **double the KV-cache capacity**, **boost long-context decode
~21%**, and **boost single-stream throughput ~42%** over a straight vLLM launch,
with quality held equal to the fp8-KV baseline.

This repo is the **build** — a `Dockerfile`, the kernel/enablement `mods/`, an
example multi-node recipe, and docs — so anyone with the same hardware can
reproduce it. The built multi-GB image itself is **not** in git (see
[`.gitignore`](.gitignore)); build it locally and optionally push it to a
registry such as GHCR (see the [Dockerfile](Dockerfile)).

> **Honesty / methodology note.** Every number below was *measured* on the actual
> 2x GB10 cluster (interleaved A/B, warm, multiple reps), not estimated from a
> roofline. Several plausible-sounding "wins" were *disproven* by measurement and
> are reported as such — see [What did NOT help](#what-did-not-help). The guiding
> rule throughout was *measure, don't assert*.

> **Authorship.** The kernels, the empirical lever analysis, and this writeup were
> written and built by **Claude** (Anthropic, via Claude Code), on top of the
> projects credited below. See [Credits, sources & authorship](#credits-sources--authorship).

---

## Quick start (pull & run)

The image is **self-contained**: pull it, set the **node role + peer IP**, run.
The head node forms the Ray cluster and serves vLLM with the optimized Config C;
**the model auto-downloads on first start**. Full commands (head, worker,
single-node, curl test) are in **[QUICKSTART.md](QUICKSTART.md)**.

```bash
# node 1 (worker) — start first:
docker run -d --name mimo-worker --gpus all --network host --ipc=host \
  -v hf-cache:/root/.cache/huggingface \
  -e NODE_ROLE=worker -e HEAD_ADDR=<HEAD_IP> \
  ghcr.io/<OWNER>/mimo-spark-optimized:latest

# node 0 (head) — the endpoint on :8000:
docker run -d --name mimo-head --gpus all --network host --ipc=host \
  -v hf-cache:/root/.cache/huggingface \
  -e NODE_ROLE=head -e HEAD_ADDR=<HEAD_IP> -p 8000:8000 \
  ghcr.io/<OWNER>/mimo-spark-optimized:latest
```

Replace `<OWNER>` (your registry namespace) and `<HEAD_IP>` (node 0's IP). Mount
a **persistent volume** at the HF cache — the NVFP4 weights are ~171 GB and each
node needs its own copy. See **[QUICKSTART.md](QUICKSTART.md)** for the curl test,
the single-node (solo) variant, tunables, and troubleshooting.

> You still need to **build the image** from this repo (it is multi-GB and not in
> git) and push it to your registry first — see the [Dockerfile](Dockerfile).

---

## TL;DR — the contributions

| # | Contribution | Measured effect |
|---|--------------|-----------------|
| a | **4-bit nvfp4 KV-cache** custom store/decode on MiMo's DiffKV | ~**2x KV capacity** (e.g. ~470–537K tokens vs ~233K fp8); quality neutral vs fp8 |
| b | **WMMA tensor-core flash-decode** kernel (cudagraph-capturable, MTP-aware) | **+21%** long-context decode (≈ +59% on the full-attn portion → 4x attn) |
| c | **`NCCL_CROSS_NIC` multi-rail** over the 2 RoCE NICs | **+42%** single-stream, **+83%** @C=8 — the single biggest speed lever |
| d | **`cudagraph_capture_sizes` boot limit** `[1,2,4,8]` | **~7x faster boot** (≈28 min → ~4 min), applies to every restart |
| e | **`o_proj` → MXFP8** selective re-quant | **+8.5% KV** (quality held); **no** speed change |
| f | **`lm_head` → NVFP4 (W4A16)** Marlin weight-only FP4 | **+27% tok/s on tool/code** (~49 tok/s), quality-neutral (56/79); +2% KV |

**The empirical lesson:** single-stream decode on TP=2 here is **latency /
communication-bound**, *not* weight-bandwidth-bound. That is why `NCCL_CROSS_NIC`
(communication) helped a lot while weight quantization (`o_proj`→MXFP8) gave +0%
tok/s — it only bought KV capacity. The throughput ceiling is reached at
**concurrency**, not single stream: aggregate ~**418 tok/s @C=24** (~13x single).

---

## The model & why it's hard on 2x GB10

MiMo-V2.5-NVFP4 is ~151 GB of NVFP4 expert weights — it does **not** fit on one
GB10 (128 GB unified memory), so it runs TP=2 across two Sparks over RoCE. GB10 is
a *unified-memory* device (CPU+GPU share 128 GB), and the CUDA allocator does not
OOM — it can fill RAM→swap→**freeze** the box, which shapes the whole deployment
(see [`docs/MULTI_NODE_SETUP.md`](docs/MULTI_NODE_SETUP.md)).

### Architecture facts that drove the kernels

- **48 transformer layers**: **9 full-attention** + **39 sliding-window
  attention (SWA)**, window 128.
- **DiffKV head dims**: K head_dim **192**, V head_dim **128** (asymmetric).
- **MoE**: 256 experts, **top-8** routing; ~15B active params/token.
- Per-rank under TP=2: 32 q-heads / 2 kv-heads → **G = 16** query-heads-per-kv
  (this exactly fills the WMMA M=16 tile — important for kernel (b)).
- **MTP** (Multi-Token Prediction, `num_speculative_tokens=2`) is on: each decode
  step has `q_len=3` per sequence (draft+verify), which the decode kernel must
  handle.

These shapes are *hard-wired assumptions* in the kernels — they are MiMo-V2.5 /
GB10 / TP=2 specific.

---

## The contributions in detail

### (a) 4-bit nvfp4 KV-cache  →  ~2x capacity, quality-neutral

vLLM's DiffKV backend refuses any quantized KV-cache. The custom mod
([`mods/nvfp4-kv-diffkv`](mods/nvfp4-kv-diffkv)) adds an end-to-end **nvfp4 (4-bit)
KV path** on MiMo's asymmetric K192/V128 DiffKV:

- **Store**: `nvfp4_kv_quantize` (global-scale `gs=1.0`) + a scatter into a packed
  per-token layout `[k_fp4 96 | k_bsf 12 | v_fp4 64 | v_bsf 8] = 180 B` (vs fp8
  320 B → **1.78x** byte ratio).
- **Decode**: paged gather via the block-table → **inline Triton dequant** in the
  fused attention kernel (uint32-vectorized nibble unpack + e2m1 + fp8 block-scale,
  validated **bit-exact** vs FlashInfer, rel-err 0.0).
- A subtle global-scale convention bug (the KV quantizer wants a *small* global
  scale, inverse to the weight `nvfp4_quantize`) and three vLLM-internal DiffKV
  plumbing gaps (stale `quant_mode`, stale cache dtype, shape-based store gating)
  were fixed in the mod.

**Result:** KV pool e.g. **~317K tokens @ util0.85/mml64K** (concurrency 4.85x) vs
~178K fp8 same config; **~470–537K** at the deployment config. Quality holds:
needle-in-haystack retrieval **exact at 16K/32K/48K/75K/100K/150K**, coherence
clean. On reliability-bench (79 tasks) the KV-sensitive hallucination/recall tasks
are **identical to fp8** (existence_check 24/24 = 24/24, refusal 9/9 = 9/9); the
~12%/layer attention reconstruction error from 4-bit KV does **not** degrade real
output.

### (b) WMMA tensor-core flash-decode  →  +21% long-context decode

A naive per-nibble dequant attention is ~4x slower than bf16, which long looked
like a "wall". It wasn't. The custom **WMMA flash-decode kernel**
([`mods/nvfp4-kv-diffkv/wmma_decode.py`](mods/nvfp4-kv-diffkv/wmma_decode.py)) uses
`mma.sync` tensor-cores for *both* matmuls (S = Q·Kᵀ and acc = P·V), with the
fp4→bf16 dequant in shared memory, GQA batched at M=16, DiffKV K192/V128, split-K +
online softmax:

| context | WMMA-paged | Triton-nvfp4 | bf16 | WMMA/Triton |
|--------:|-----------:|-------------:|-----:|------------:|
| 16K     | 1479 µs    | 2605 µs      | 706  | **1.76x**   |
| 64K     | 4994 µs    | 11553 µs     | 2784 | **2.31x**   |
| 131K    | 9777 µs    | 23114 µs     | 5571 | **2.36x**   |

(rel-err 0.0026 vs Triton — just dequant-rounding between two correct nvfp4 impls.)
This halves the nvfp4-KV long-context decode penalty from ~4.2x to ~1.9x of bf16.

Getting it to actually fire in-engine took fixing a full bug stack: G=8→**16**,
env not propagating to Ray workers, full-attn layers using **block_size 64** (not
32), **MTP q_len=3** expansion, a prefill gate, and — the real killer — a **CUDA
stream mismatch** (the kernel launched on the default stream while torch recycled
its scratch on another, corrupting state between layer launches; fix:
`getCurrentCUDAStream()` on every launch). It is **gated** (`VLLM_WMMA_DECODE`),
**falls back to Triton** for SWA / prefill / unsupported shapes, and is
**cudagraph-capturable** (fixed NSPLIT, static scratch).

**End-to-end:** ~**+21%** long-context decode (5.94 vs ~3–4 tok/s @74K with
cudagraph), or measured **4.90 vs 3.09 tok/s @100K = +59%** streaming. The gain is
bounded because the 9 full-attn layers (where the kernel applies) are only part of
the MoE+39-SWA-dominated decode — but it is real, free for the 4-bit-KV config,
and **0 downside** (correct everywhere via fallback). Needle@52K stays exact.

### (c) `NCCL_CROSS_NIC=1` multi-rail  →  +42% single-stream

The two GB10s have two RoCE NICs. Using both as a multi-rail NCCL link
(`NCCL_CROSS_NIC=1`, plus `RAY_memory_monitor_refresh_ms=0` for false-OOM
protection) gave **+42% single-stream (31→44 tok/s)** and **+83% @C=8 (64→117)**.
This was the single biggest lever — and it directly proves the TP=2 single-stream
bottleneck is the inter-node all-reduce (comm/latency), not weight bandwidth.

### (d) `cudagraph_capture_sizes` boot limit  →  ~7x faster boot

Limiting cudagraph capture to `[1,2,4,8]` cut cold boot from ~**28 min to ~4 min**
(~7x). This also applies to every production restart.

### (e) `o_proj` → MXFP8 selective re-quant  →  +8.5% KV (no speed)

MiMo keeps `o_proj`/`lm_head` in BF16. Re-quantizing per-layer `o_proj` to MXFP8
(calibration-free, same format the checkpoint already uses for `qkv`) frees ~1.6 GB
→ **+8.5% KV** (434K vs 400K), quality held. It gives **+0% tok/s** — confirming
the latency/comm-bound finding. Tooling:
[`tools/quantize_oproj_mxfp8.py`](tools/quantize_oproj_mxfp8.py).

### (f) `lm_head` → NVFP4 (W4A16)  →  +27% tok/s on tool/code, quality-neutral

MiMo's `lm_head` (BF16, `[152576, 4096]`, ~1.25 GB) is read ~3x per decode step
(once for the main verify + twice for the MTP drafter). Quantizing it to **NVFP4
W4A16** (4-bit weights, BF16 activations, Marlin weight-only FP4 GEMM — a *true*
4-bit read, not dequant-to-BF16 emulation) cuts that traffic ~4x. Standalone the
lm_head GEMM is ~5x faster at decode batch; end-to-end:

| regime | bf16 lm_head | **nvfp4 lm_head** | Δ |
|--------|-------------:|------------------:|---|
| echo (code/tool reproduction) | 38.8 | **49.3** | **+27%** (step 71→57 ms) |
| novel (control) | 44.1 | 46.9 | +6% |
| diverse (prose) | 36.4 | 35.8 | ~wash |

The win lands on **tool/code-heavy** decode; pure prose is ~neutral because the
MTP drafter now also uses the quantized head, slightly lowering its acceptance
and offsetting the step-time gain. **Quality is neutral**: reliability-bench
**56/79 = identical to the bf16 baseline**, with the KV/recall-sensitive tasks
(existence 24/24, refusal-to-fabricate 9/9) perfect. Plus ~+2% KV. lm_head is the
most logit-sensitive layer, so this was verified, not assumed.

Implementation ([`mods/fix-lmhead-nvfp4`](mods/fix-lmhead-nvfp4),
[`tools/quantize_lmhead_nvfp4.py`](tools/quantize_lmhead_nvfp4.py)) needed six
vLLM-wiring fixes: `get_quant_method` routing for `ParallelLMHead`; leaf-match for
the Omni `language_model.lm_head` prefix; a *filtered* checkpoint shard so the
BF16 head isn't double-loaded over the packed one; `weight_scale_2` shape `[1]`;
`params_dtype` on the embedding (Marlin reads it); and the MTP drafter head
getting its own `quant_config`.

---

## Benchmarks

All on 2x DGX Spark (GB10), MiMo-V2.5-NVFP4, TP=2, MTP-2, the deployment config
("Config C" = 4-bit KV + cudagraph + WMMA + NCCL_CROSS_NIC).

### Throughput (tok/s)

| Streams (C) | tok/s (aggregate), `max_num_seqs=32` |
|------------:|------------------:|
| 1           | ~43–44            |
| 6           | ~112              |
| 16          | ~268              |
| 24          | ~402              |
| **32**      | **~493**          |

**`max_num_seqs` is the multi-stream dial.** The default (6) caps aggregate
throughput at ~112 tok/s — every stream past 6 just queues. Raising it to 32
unlocks near-linear scaling to **~493 tok/s** (≈ 4.4x, ~15 tok/s/stream sustained).
The **4-bit KV-cache is what makes this affordable** (the pool is still ~372K
tokens at `max_num_seqs=32`). Trade-off: more concurrent slots = less *guaranteed*
KV per stream — so it's a regime choice, **max-KV-per-request (low mns, long
single contexts)** vs **max-aggregate (high mns)**. Set it via `MAX_NUM_SEQS`
(see [QUICKSTART.md](QUICKSTART.md)). Streams ≤6 are MoE-arithmetic-limited
(small-batch expert under-utilization), not mns-limited.

Single-stream decode tok/s is **content-dependent** via MTP acceptance: structured
output (lists/code echo) ~34 tok/s @89% accept, factual ~25 @55%, creative prose
~23 @41%. The GB10 read-bandwidth floor (~233 GB/s, ~4.1 GB/node/step) implies a
~119 tok/s MTP single-stream ceiling; measured ~25–44 is the latency/comm + M=1
GEMM-efficiency gap, **not** something a MoE-tile kernel can close (proven by
FlashInfer autotune giving 0% and a head-to-head vs NVIDIA's native sm121 `b12x`
kernel — CUTLASS already runs at ~71% BW at M=1 and wins everywhere).

### KV-cache capacity & quality

| Config           | KV pool (tokens)        | reliability-bench (79) |
|------------------|-------------------------|------------------------|
| fp8 KV (baseline)| ~233K @ util0.89        | 56/79                  |
| **4-bit nvfp4 KV** | **~470–537K** (2x)    | 53/79 (≈ equal)        |
| + o_proj MXFP8   | +8.5% on top            | quality held           |

The 3-point reliability delta is from the reasoning mode used in that run, **not**
KV precision: the KV-sensitive tasks (existence_check, refusal-to-fabricate) are
**identical** to fp8. 4-bit KV does not degrade quality.

---

## Hardware requirements

- **2x NVIDIA DGX Spark** (GB10, `sm_121`, ARM aarch64), connected by **2 RoCE
  NICs** (MTU 9000 / PFC recommended).
- **CUDA 13.x**, a vLLM base built for `TORCH_CUDA_ARCH_LIST=12.1a` with the
  DiffKV attention backend + ModelOpt MIXED_PRECISION path, and FlashInfer with
  the sm12x cutlass NVFP4 MoE kernels. See the [Dockerfile](Dockerfile) header for
  the exact base-image requirements (the base image is **not** published here —
  you build it).
- The MiMo-V2.5-NVFP4 checkpoint from HuggingFace (`lukealonso/MiMo-V2.5-NVFP4`).

---

## Limitations & what did NOT help

This stack is **strongly `sm_121` / 2-node / MiMo-V2.5 specific**. The kernels
hard-code MiMo shapes (K192/V128, G=16, block_size 32/64) and JIT-compile for
`sm_121a` — they will not run elsewhere unchanged.

### What did NOT help <a id="what-did-not-help"></a> (measured, reported honestly)

- **Weight quantization for speed** — `o_proj`→MXFP8 gave +0% tok/s (decode is
  latency/comm-bound, not weight-BW-bound). Value is KV capacity only.
- **MoE-backend / FlashInfer autotune / GEMM tactics** — 0% (decode MoE-GEMM at
  M=1 is memory-bandwidth-bound, not tactic-bound; CUTLASS already ~71% BW).
- **Custom MoE / scalar-CUDA attention kernels** — could not beat the existing
  tensor-core paths; a true max attention kernel needs hand-`mma.sync` PTX
  (multi-week effort).
- **Expert Parallelism (`--enable-expert-parallel`, EP=2)** — measured *worse* on
  this 2-node setup: single-stream 36 vs 43 (the all-to-all dispatch adds overhead
  with only 2 ranks). The literature's "EP helps" applies to DP=8+EP at ≥512
  concurrent requests, not 2 Sparks. **Plain TP=2 stays the winner here.** (The
  real multi-stream lever is `max_num_seqs`, above — not EP.)
- **`lm_head` → MXFP8** — superseded. The lm_head is now quantized to **NVFP4
  W4A16** instead (a true 4-bit-weight Marlin GEMM, which MXFP8 emulation could
  not deliver speed-wise). It is a **win, not a limitation** — see contribution
  (f) above (+27% tok/s on tool/code, quality-neutral 56/79). The MXFP8 route was
  abandoned (NVFP4 won).
- **NCCL_NTHREADS / inductor fusion passes / async-scheduling** — `NCCL_NTHREADS`
  was within noise; the `fuse_act_quant`/`fuse_norm_quant` passes gave 0% on
  `sm_121` (pattern-match miss) at a memory cost; `--async-scheduling` is
  incompatible with the Ray multi-node executor.
- **NCCL_NTHREADS / fusion passes / mnbt / block-size** — no reproducible gain.
- **SWA fp8 / full nvfp4 KV hybrid** — vLLM uses a uniform page size; the
  nvfp4:fp8 ratio (1.777) is never integral, so padding eats the capacity.
- **`async-scheduling`** — incompatible with the Ray executor here.

### Stacks that did NOT work at all on sm_121 (for reference)

FlashAttention-3 (sm_121 has only FA2 → PR #41797 needed for sinks), the official
cu130 cluster recipe, naive Flash-DiffKV, and the plain Triton-DiffKV without the
fixes here all failed or produced garbage on GB10.

---

## Repo layout

```
.
├── README.md                              # this file
├── QUICKSTART.md                          # "pull & run" head/worker/solo commands
├── Dockerfile                             # self-contained image: mods + entrypoint
├── docker-entrypoint.sh                   # pull&run entrypoint (Ray + Config-C serve)
├── LICENSE                                # MIT
├── .gitignore                             # excludes weights / *.so / logs / secrets
├── mods/
│   ├── fix-mimo-v2-vllm/                  # MiMo-V2.5 enablement (chat template,
│   │                                      #   MTP, tool/reasoning parser, PR #41797,
│   │                                      #   DiffKV quant-KV gate)
│   ├── fix-modelopt-mixed-mxfp8/          # ModelOpt MIXED_PRECISION MXFP8 dispatch
│   ├── nvfp4-kv-diffkv/                    # 4-bit nvfp4 KV store/decode + WMMA kernel
│   │   ├── run.sh
│   │   ├── triton_attn_diffkv.py          # DiffKV backend + capturable nvfp4 store
│   │   ├── triton_unified_attention_diffkv.py  # inline nvfp4 dequant fused attn
│   │   └── wmma_decode.py                  # WMMA tensor-core flash-decode kernel
│   └── fix-lmhead-nvfp4/                   # lm_head → NVFP4 W4A16: get_quant_method
│       └── run.sh                          #   routing + params_dtype + MTP-drafter patch
├── recipes/
│   └── mimo-v2.5-nvfp4.example.yaml        # Config C launch recipe (placeholders)
├── docs/
│   └── MULTI_NODE_SETUP.md                 # 2-node Ray + RoCE/NCCL setup guide
└── tools/
    ├── quantize_oproj_mxfp8.py             # selective o_proj→MXFP8 re-quant (CPU)
    ├── quantize_lmhead_nvfp4.py            # lm_head → NVFP4 W4A16 overlay (CPU)
    └── README.md
```

Each mod is a directory with a `run.sh` that patches the installed vLLM in place
and self-validates. The [Dockerfile](Dockerfile) applies them at build time.

## Credits, sources & authorship

**Authorship.** The custom kernels (the 4-bit nvfp4 KV-cache + cudagraph-capturable
store, the WMMA tensor-core flash-decode kernel, the `o_proj`/`lm_head` MXFP8
selective re-quant tool, and the `ParallelLMHead` MXFP8 dispatch patch), the
empirical lever analysis, and this documentation were **written and built by
Claude** — Anthropic's Claude, via Claude Code — during a hands-on optimization
session directly on the target 2x GB10 hardware. The work stands on top of the
foundations below, which deserve the credit for making it possible.

**Model & weights**
- **MiMo-V2.5** — [Xiaomi / XiaomiMiMo](https://huggingface.co/XiaomiMiMo) (the base model).
- **NVFP4 checkpoint** — [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4).

**Inference stack**
- [**vLLM**](https://github.com/vllm-project/vllm) — the engine these mods/kernels patch.
- [**FlashInfer**](https://github.com/flashinfer-ai/flashinfer) — NVFP4 quantize/GEMM primitives the kernels build on.
- The triton DiffKV attention path builds on **vLLM PR #41797** (triton_attn DiffKV — attention sinks without FA3, needed on `sm_121` which only has FA2).
- The MiMo-V2.5 vLLM enablement (chat template, MTP, tool/reasoning parsers) is based on the community **spark-vllm recipe (eugr PR #251)**.

**Key lever — `NCCL_CROSS_NIC` multi-rail (+42% single-stream)**
- Surfaced from public **2-3x DGX-Spark community deployment work** (notably the
  tonyd2wild *MiMo-V2.5-Omni 3x-DGX-Spark* repo). This single env flag was the
  biggest single-stream win and is easy to miss — full credit to that community work.

**Hardware & systems docs**
- **NVIDIA DGX Spark (GB10)**, NCCL, RoCEv2 — NVIDIA documentation.

> Where a number or technique came from external work it is cited inline above and
> in the relevant section; everything else (the kernels and the measured analysis)
> was produced by Claude in-session.

## License

MIT — see [LICENSE](LICENSE). Provided as-is; the kernels are research-grade and
specific to this hardware.

## Update 2026-06 — NSPLIT 1-wave tuning + async scheduling unlock

Two further **lossless** wins on top of the table above, both validated live on 2× GB10 (TP=2, Ray, MiMo-V2.5-NVFP4 + MTP2):

### (g) NSPLIT 1-wave tuning  →  +7–15% decode attention, bit-exact

The TK decode kernel's split-K count is capped so total blocks land on **one full GPU wave**
— `num_kv_heads × NSPLIT × num_seqs ≈ 192` (4 blocks/SM × 48 SMs) — instead of the previous
~2.7 waves (NSPLIT=256 → 512 blocks → wave-quantization waste). One-line change in
`tk_decode.py` (`_nsplit` cap `256→96`). Isolated kernel: **+7–15% at 32K–100K** context,
rel-err ~5e-3 (bf16 reduction reorder, no systematic error). End-to-end it is marginal because
decode is MoE-dominated, but it is free and bit-stable.

### (h) async scheduling for MTP on Ray  →  lossless +6–8% tok/s  (`mods/enable-async-mtp/`)

vLLM disables async scheduling for MTP speculative decoding — but this is a conservative
oversight, **not** a technical limitation:

* The Ray distributed executor already implements async execution
  (`execute_model(non_block=True) → FutureWrapper`, `max_concurrent_batches=2`) but never
  overrides `supports_async_scheduling()` (inherits the base `False`).
* `vllm/config/vllm.py` exempts `"draft_model"` from the async-disable gates but **forgets
  `"mtp"`** — even though the hard-fail error message itself lists *"EAGLE/MTP/Draft
  Model/NGram"* as supported.

The mod patches both (Ray flag + the two `mtp` exemptions); launch with `--async-scheduling`.
Async is lossless by construction (identical tokens; it only overlaps CPU scheduling with GPU
execution). On a GPU-bound decode (~90% util) it recovers the CPU-scheduling gap:

| context | before | async |
|---|---|---|
| short, single-stream | 36.7 tok/s | **39.2** (+7%) |
| 64K decode | 9.8 | **10.6** (+8%) |
| 100K decode | 8.3 | **8.8** (+6%) |

Quality unchanged (needle recall = dense baseline). This is the highest-leverage item for
anyone running MiMo (or any MTP model) multi-node on Ray.

> Tried and rejected (documented for honesty): KV-sparsity (Quest-style) — block-level
> criticality is too loose at head_dim 192 to preserve mid-context needle recall; a custom
> TK **prefill** kernel — correct but ~11–16% slower than the already tensor-core-efficient
> Triton prefill (decode is GEMV where Triton is weak, prefill is GEMM where it is not).
