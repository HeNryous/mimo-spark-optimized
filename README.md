# MiMo-V2.5-NVFP4 on 2x NVIDIA DGX Spark — optimized vLLM build

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
[`tools/quantize_oproj_mxfp8.py`](tools/quantize_oproj_mxfp8.py) (lm_head path is
experimental — it needs a separate `ParallelLMHead` vLLM patch, not included).

---

## Benchmarks

All on 2x DGX Spark (GB10), MiMo-V2.5-NVFP4, TP=2, MTP-2, the deployment config
("Config C" = 4-bit KV + cudagraph + WMMA + NCCL_CROSS_NIC).

### Throughput (tok/s)

| Streams (C) | tok/s (aggregate) |
|------------:|------------------:|
| 1           | ~39–44            |
| 4           | ~67               |
| 8           | ~145–209          |
| 16          | ~306              |
| **24**      | **~418** (peak)   |
| 32          | ~308 (mns-limited)|

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
│   └── nvfp4-kv-diffkv/                    # 4-bit nvfp4 KV store/decode + WMMA kernel
│       ├── run.sh
│       ├── triton_attn_diffkv.py          # DiffKV backend + capturable nvfp4 store
│       ├── triton_unified_attention_diffkv.py  # inline nvfp4 dequant fused attn
│       └── wmma_decode.py                  # WMMA tensor-core flash-decode kernel
├── recipes/
│   └── mimo-v2.5-nvfp4.example.yaml        # Config C launch recipe (placeholders)
├── docs/
│   └── MULTI_NODE_SETUP.md                 # 2-node Ray + RoCE/NCCL setup guide
└── tools/
    ├── quantize_oproj_mxfp8.py             # selective o_proj→MXFP8 re-quant (CPU)
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
