# =============================================================================
# MiMo-V2.5-NVFP4 optimized vLLM image for 2x NVIDIA DGX Spark (GB10, sm_121)
#
# This image takes a working sm_121 vLLM base image and bakes in the MiMo-V2.5
# enablement + the custom kernel mods (4-bit nvfp4 KV-cache, WMMA tensor-core
# flash-decode, ModelOpt mixed-precision MXFP8 dispatch fix).
#
# The mods are directories under mods/, each with a run.sh that patches the
# installed vLLM site-packages in place. Applying them at BUILD time bakes the
# changes into the image so they are already present at container start.
#
# ---------------------------------------------------------------------------
# BASE_IMAGE requirements (build this yourself; it is NOT published here):
#   * vLLM built for CUDA 13.x, TORCH_CUDA_ARCH_LIST="12.1a" (GB10 = sm_121a),
#     ARM aarch64. A recent main-branch vLLM works; this stack was developed
#     against a late-2025/early-2026 vLLM main snapshot. The DiffKV attention
#     backend (triton_attn_diffkv) and ModelOpt MIXED_PRECISION quant path must
#     be present (they are on recent main).
#   * FlashInfer with the sm12x cutlass NVFP4 MoE kernels
#     (gen_cutlass_fused_moe_sm120_module / b12x), built for arch 12.1a.
#   * Python 3.12, dist-packages at /usr/local/lib/python3.12/dist-packages
#     (the mod run.sh scripts assume this path).
#
# vLLM PR #41797 (TRITON_ATTN_DIFFKV sinks without FA3) is required on sm_121,
# which only ships FlashAttention-2. The fix-mimo-v2-vllm mod fetches+applies it
# at build time if it is not already in the base.
#
# Build:
#   docker build --build-arg BASE_IMAGE=vllm-base:sm121 -t mimo-spark-optimized .
#
# Optionally push the built image to a registry (e.g. GHCR):
#   docker tag mimo-spark-optimized ghcr.io/<you>/mimo-spark-optimized:latest
#   docker push ghcr.io/<you>/mimo-spark-optimized:latest
# =============================================================================

# Replace with your own sm_121 vLLM base image.
ARG BASE_IMAGE=vllm-base:sm121

FROM ${BASE_IMAGE} AS optimized

ARG SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages

# Tools used by the mods (patch fetching, ast/py_compile checks). Most bases
# already have these; the `|| true` keeps the build going on minimal bases.
RUN (command -v curl >/dev/null 2>&1 && command -v git >/dev/null 2>&1) || \
    (apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates \
     && rm -rf /var/lib/apt/lists/*) || true

# Copy the mods into the image.
WORKDIR /opt/mimo-mods
COPY mods/ ./mods/

# Apply the mods in order. Each run.sh patches the installed vLLM in place and
# self-validates (ast.parse / py_compile / import checks). The build fails loudly
# if any anchor is missing (e.g. base vLLM too old/new).
#   1. fix-mimo-v2-vllm          : MiMo-V2.5 enablement (chat template -> /root,
#                                  MTP, tool/reasoning parser, DiffKV quant-KV gate,
#                                  vLLM PR #41797)
#   2. fix-modelopt-mixed-mxfp8  : ModelOpt MIXED_PRECISION MXFP8 dispatch
#   3. nvfp4-kv-diffkv           : 4-bit nvfp4 KV store/decode + WMMA decode kernel
RUN set -eux; \
    for mod in fix-mimo-v2-vllm fix-modelopt-mixed-mxfp8 nvfp4-kv-diffkv; do \
        echo "=== applying mod: $mod ==="; \
        chmod +x "mods/$mod/run.sh"; \
        "mods/$mod/run.sh"; \
    done

# Quick import smoke-test of the patched modules.
RUN python3 -c "import ast; \
    ast.parse(open('${SITE_PACKAGES}/vllm/v1/attention/backends/triton_attn_diffkv.py').read()); \
    print('triton_attn_diffkv backend OK')"

# The MiMo chat template is written to /root/mimo_chat_template.jinja by the
# fix-mimo-v2-vllm mod; the recipe / entrypoint point --chat-template at it.

# ---------------------------------------------------------------------------
# Self-contained "pull & run" layer: bake the launch orchestration, the
# Config-C recipe (reference), the o_proj overlay tool, and the entrypoint, so
# users only set a couple of env vars at `docker run` time.
# ---------------------------------------------------------------------------
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY recipes/mimo-v2.5-nvfp4.example.yaml /opt/mimo/recipes/mimo-v2.5-nvfp4.example.yaml
COPY tools/ /opt/mimo/tools/
COPY docs/ /opt/mimo/docs/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ---------------------------------------------------------------------------
# Config-C optimized env defaults (mirror recipes/mimo-v2.5-nvfp4.example.yaml).
# These are baked so "pull & run" reproduces the tuned deployment without the
# user re-specifying them. NODE-specific values (role, peer IP, HCA names) are
# resolved at runtime by the entrypoint — nothing host-specific is baked here.
# ---------------------------------------------------------------------------
# Architecture / allocator
ENV TORCH_CUDA_ARCH_LIST="12.1a" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    MIMO_DEFAULT_THINKING_TOKEN_BUDGET="4096"

# THE comm hammer: multi-rail NCCL over the 2 RoCE NICs (+42% single-stream).
# NCCL_IB_HCA is intentionally NOT set here — the entrypoint auto-detects it
# (ibdev2netdev) or you override it at runtime (-e NCCL_IB_HCA=...).
ENV NCCL_CROSS_NIC="1" \
    NCCL_CUMEM_ENABLE="0" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_NTHREADS="8" \
    NCCL_NSOCKS_PERTHREAD="2" \
    NCCL_BUFFSIZE="8388608" \
    NCCL_IB_DISABLE="0"

# Ray executor knobs — keep V2 / DAG-overlap OFF on this >=100GB model (extra
# unified memory -> freeze risk). Disable Ray's false-OOM monitor (unified mem).
ENV RAY_memory_monitor_refresh_ms="0" \
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND="0" \
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM="0" \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="0" \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"

# FlashInfer NVFP4 MoE backend (sm12x cutlass kernel).
ENV VLLM_NVFP4_GEMM_BACKEND="flashinfer-cutlass" \
    VLLM_USE_FLASHINFER_MOE_FP4="1" \
    VLLM_FLASHINFER_MOE_BACKEND="throughput"

# 4-bit nvfp4 KV-cache + WMMA tensor-core flash-decode (cudagraph-safe).
# Set VLLM_WMMA_DECODE=0 to fall back to Triton everywhere.
ENV VLLM_NVFP4_INLINE="1" \
    VLLM_WMMA_DECODE="1" \
    VLLM_WMMA_INSPECT="0" \
    VLLM_WMMA_COMPARE="0" \
    VLLM_WMMA_AUTHCOMPARE="0"

# Config-C serve defaults (overridable per-run). See docker-entrypoint.sh.
ENV MODEL="lukealonso/MiMo-V2.5-NVFP4" \
    TP_SIZE="2" \
    GPU_MEM_UTIL="0.87" \
    MAX_MODEL_LEN="98304" \
    MAX_NUM_SEQS="6" \
    MAX_NUM_BATCHED_TOKENS="16384" \
    SERVED_MODEL_NAME="MiMo-V2.5-NVFP4" \
    PORT="8000" \
    HOST="0.0.0.0" \
    RAY_PORT="6379" \
    HF_HOME="/root/.cache/huggingface"

EXPOSE 8000

# "Pull & run": set NODE_ROLE (+ HEAD_ADDR for multi-node) and go. The entrypoint
# forms the Ray cluster (head/worker) or runs solo, auto-downloads the model on
# first start, and serves with the Config-C arg list.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
