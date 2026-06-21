#!/usr/bin/env bash
# =============================================================================
# docker-entrypoint.sh — "pull & run" entrypoint for MiMo-V2.5-NVFP4 (Config C)
# on 2x NVIDIA DGX Spark (GB10, sm_121, TP=2) over Ray + RoCE.
#
# Pull the image, set the node role + peer IP, run. The head node forms a Ray
# cluster, waits for the worker to join, then serves vLLM with the optimized
# Config-C argument set. The model auto-downloads on first start.
#
# ---------------------------------------------------------------------------
# Runtime environment (set with `docker run -e ...`):
#
#   NODE_ROLE        head | worker | solo
#                    - head   : Ray head + `vllm serve` (rank 0). The endpoint.
#                    - worker : Ray worker that joins HEAD_ADDR and blocks.
#                    - solo   : single-node, no Ray (for a big single GPU test).
#                    If unset and TP_SIZE=1            -> solo.
#                    If unset and TP_SIZE>1            -> error (role required).
#
#   HEAD_ADDR        IP/host of the Ray head. Required for head and worker when
#                    TP_SIZE>1 (head binds it; worker dials it). e.g. 203.0.113.10
#
#   MODEL            HF model id or local path.
#                    Default: lukealonso/MiMo-V2.5-NVFP4
#   TP_SIZE          tensor-parallel size. Default: 2 (one GPU per node).
#
#   GPU_MEM_UTIL     gpu-memory-utilization. Default: 0.87 (Config C; see docs).
#   MAX_MODEL_LEN    max-model-len.          Default: 98304 (96K).
#   MAX_NUM_SEQS     max-num-seqs.           Default: 6 (raise ~24 for aggregate).
#   MAX_NUM_BATCHED_TOKENS  Default: 16384.
#   SERVED_MODEL_NAME       Default: MiMo-V2.5-NVFP4.
#   PORT                    Default: 8000.
#   HOST                    Default: 0.0.0.0.
#   RAY_PORT                Default: 6379 (Ray head GCS port).
#
#   HF_HOME          HuggingFace cache root. Default: /root/.cache/huggingface.
#                    *** Mount a persistent volume here *** or the ~171 GB of
#                    weights re-download on every fresh container. BOTH nodes
#                    need the weights present (each rank reads its own copy).
#
#   NCCL_IB_HCA      RoCE HCA list, e.g. "rocep1s0f0,rocep1s0f1". If unset, the
#                    entrypoint auto-detects via ibdev2netdev. Override here if
#                    detection is wrong or you want a subset.
#   NCCL_SOCKET_IFNAME  control-plane iface. Auto-detected if unset.
#
#   ENABLE_OPROJ_MXFP8   "1" to opt in to the o_proj->MXFP8 KV-capacity overlay
#                    (+~8.5% KV, no speed change). Default OFF (keeps pull&run
#                    simple). When on, the entrypoint generates the overlay with
#                    tools/quantize_oproj_mxfp8.py and serves the merged model.
#
#   EXTRA_VLLM_ARGS  appended verbatim to the `vllm serve` command (escape hatch).
#   SKIP_DOWNLOAD    "1" to skip the auto-download pre-check (model is present).
#
# All optimized NCCL / vLLM / WMMA env defaults are baked as ENV in the
# Dockerfile; this script only fills NODE-specific values and assembles the
# Config-C `vllm serve` argument list.
# =============================================================================
set -euo pipefail

log()  { printf '[entrypoint] %s\n' "$*" >&2; }
die()  { printf '[entrypoint][FATAL] %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Defaults (mirror recipes/mimo-v2.5-nvfp4.example.yaml — "Config C")
# ---------------------------------------------------------------------------
MODEL="${MODEL:-lukealonso/MiMo-V2.5-NVFP4}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-V2.5-NVFP4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
RAY_PORT="${RAY_PORT:-6379}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HOME

CHAT_TEMPLATE="${CHAT_TEMPLATE:-/root/mimo_chat_template.jinja}"
ENABLE_OPROJ_MXFP8="${ENABLE_OPROJ_MXFP8:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# Resolve NODE_ROLE: unset + TP=1 -> solo; unset + TP>1 -> error.
NODE_ROLE="${NODE_ROLE:-}"
if [[ -z "$NODE_ROLE" ]]; then
    if [[ "$TP_SIZE" == "1" ]]; then
        NODE_ROLE="solo"
        log "NODE_ROLE unset and TP_SIZE=1 -> defaulting to solo (single-node) mode."
    else
        die "NODE_ROLE is required for TP_SIZE>1. Set -e NODE_ROLE=head (the endpoint) on node 0 and -e NODE_ROLE=worker on node 1, plus -e HEAD_ADDR=<HEAD_IP> on both. (Or set TP_SIZE=1 for solo mode.)"
    fi
fi
case "$NODE_ROLE" in
    head|worker|solo) ;;
    *) die "NODE_ROLE must be one of: head | worker | solo (got '$NODE_ROLE')." ;;
esac

# ---------------------------------------------------------------------------
# Preflight: GPU present (skip with ALLOW_NO_GPU=1 for dry inspection).
# ---------------------------------------------------------------------------
if [[ "${ALLOW_NO_GPU:-0}" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        die "No GPU visible (nvidia-smi failed). Run the container with '--gpus all'. (Set ALLOW_NO_GPU=1 only for non-GPU inspection.)"
    fi
fi

# ---------------------------------------------------------------------------
# NCCL HCA / socket-iface auto-detection (override via env). Mirrors the
# ibdev2netdev approach used by the reference launcher: every IB device that is
# 'Up' becomes an HCA, and its first netdev with an IP becomes the socket iface.
# No hostnames or HCA names are hardcoded.
# ---------------------------------------------------------------------------
autodetect_fabric() {
    command -v ibdev2netdev >/dev/null 2>&1 || { log "ibdev2netdev not found; skipping fabric auto-detect (set NCCL_IB_HCA/NCCL_SOCKET_IFNAME manually if NCCL falls back to sockets)."; return 0; }

    # Lines look like: "<ibdev> port 1 ==> <netdev> (Up)"
    local pairs hca_list="" sock_if=""
    pairs="$(ibdev2netdev 2>/dev/null | awk '/Up\)/{print $1" "$5}')" || true
    [[ -z "$pairs" ]] && { log "No 'Up' IB devices found; leaving NCCL fabric vars unset."; return 0; }

    local ibdev netdev
    while read -r ibdev netdev; do
        [[ -z "$ibdev" ]] && continue
        hca_list="${hca_list:+$hca_list,}$ibdev"
        if [[ -z "$sock_if" ]] && ip addr show "$netdev" 2>/dev/null | grep -q 'inet '; then
            sock_if="$netdev"
        fi
    done <<< "$pairs"

    if [[ -z "${NCCL_IB_HCA:-}" && -n "$hca_list" ]]; then
        export NCCL_IB_HCA="$hca_list"
        log "Auto-detected NCCL_IB_HCA=$NCCL_IB_HCA"
    fi
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" && -n "$sock_if" ]]; then
        export NCCL_SOCKET_IFNAME="$sock_if"
        log "Auto-detected NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    fi
}
autodetect_fabric

# ---------------------------------------------------------------------------
# Model resolution + auto-download. If MODEL is a local path, use it directly.
# Otherwise download the HF snapshot into HF_HOME (persistent volume strongly
# recommended) and serve from the resolved snapshot directory.
# ---------------------------------------------------------------------------
resolve_model_path() {
    if [[ -d "$MODEL" ]]; then
        RESOLVED_MODEL="$MODEL"
        log "MODEL is a local directory: $RESOLVED_MODEL"
        return 0
    fi

    if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
        log "SKIP_DOWNLOAD=1 — assuming '$MODEL' is already cached; passing the id straight to vLLM."
        RESOLVED_MODEL="$MODEL"
        return 0
    fi

    log "Ensuring model '$MODEL' is present in HF cache ($HF_HOME) ..."
    log "  (first start downloads ~171 GB — mount a persistent volume at \$HF_HOME to avoid re-downloads; BOTH nodes need the weights)."

    # Prefer the modern `hf download`; fall back to `huggingface-cli download`.
    local dl_ok=0
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL" >/dev/null && dl_ok=1 || dl_ok=0
    fi
    if [[ "$dl_ok" != "1" ]] && command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL" >/dev/null && dl_ok=1 || dl_ok=0
    fi
    if [[ "$dl_ok" != "1" ]]; then
        # Last resort: python API.
        python3 - "$MODEL" <<'PY' && dl_ok=1 || dl_ok=0
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
PY
    fi
    [[ "$dl_ok" == "1" ]] || die "Failed to download model '$MODEL'. Check network / HF availability, or pre-populate the \$HF_HOME volume."

    # Resolve the on-disk snapshot directory so vLLM loads from local files.
    RESOLVED_MODEL="$(python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PY
)"
    [[ -n "$RESOLVED_MODEL" && -d "$RESOLVED_MODEL" ]] || die "Could not resolve local snapshot path for '$MODEL' after download."
    log "Model snapshot resolved at: $RESOLVED_MODEL"
}

# ---------------------------------------------------------------------------
# Optional o_proj -> MXFP8 overlay (opt-in; default OFF). Generates a small
# merged checkpoint that re-quantizes o_proj (and lm_head) to MXFP8 for ~+8.5%
# KV capacity, then serves *that* directory instead of the base snapshot.
# ---------------------------------------------------------------------------
maybe_build_oproj_overlay() {
    [[ "$ENABLE_OPROJ_MXFP8" == "1" ]] || return 0
    local out="${OPROJ_OUT_DIR:-$HF_HOME/mimo-oproj-mxfp8-overlay}"
    if [[ -f "$out/config.json" ]]; then
        log "o_proj->MXFP8 overlay already present at $out — reusing."
    else
        log "ENABLE_OPROJ_MXFP8=1 — generating o_proj->MXFP8 overlay (CPU) into $out ..."
        python3 /opt/mimo/tools/quantize_oproj_mxfp8.py \
            --snapshot "$RESOLVED_MODEL" --out "$out" \
            --oproj-format mxfp8 --lmhead-format mxfp8 \
            || die "o_proj->MXFP8 overlay generation failed."
    fi
    RESOLVED_MODEL="$out"
    log "Serving o_proj->MXFP8 overlay: $RESOLVED_MODEL"
}

# ---------------------------------------------------------------------------
# Assemble the Config-C `vllm serve` argument list (head/solo only).
# ---------------------------------------------------------------------------
build_vllm_cmd() {
    VLLM_ARGS=(
        serve "$RESOLVED_MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --trust-remote-code
        --dtype auto
        --kv-cache-dtype nvfp4
        --block-size 32
        --attention-backend triton_attn_diffkv
        --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
        --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}'
        --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}'
        --host "$HOST"
        --port "$PORT"
        --tensor-parallel-size "$TP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
        --no-async-scheduling
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --enable-prefix-caching
        --enable-chunked-prefill
        --load-format instanttensor
        --enable-auto-tool-choice
        --tool-call-parser mimo
        --reasoning-parser mimo
        --chat-template "$CHAT_TEMPLATE"
    )
    # Multi-node uses Ray; solo runs in-process.
    if [[ "$NODE_ROLE" == "head" ]]; then
        VLLM_ARGS+=( --distributed-executor-backend ray )
    fi
    # Append the user escape-hatch args (word-split intentionally).
    if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
        # shellcheck disable=SC2206
        local extra=( $EXTRA_VLLM_ARGS )
        VLLM_ARGS+=( "${extra[@]}" )
    fi
}

# ---------------------------------------------------------------------------
# Ray head: start GCS, wait for the expected number of nodes, then serve.
# ---------------------------------------------------------------------------
run_head() {
    [[ -n "${HEAD_ADDR:-}" ]] || die "NODE_ROLE=head requires HEAD_ADDR (this node's reachable IP, which workers dial). e.g. -e HEAD_ADDR=<HEAD_IP>"

    local expected_nodes=$(( TP_SIZE * PP_SIZE ))
    log "Starting Ray HEAD on $HEAD_ADDR:$RAY_PORT (expecting $expected_nodes node(s) total) ..."
    ray start --head \
        --node-ip-address="$HEAD_ADDR" \
        --port="$RAY_PORT" \
        --num-gpus=1 \
        --include-dashboard=false \
        --disable-usage-stats

    log "Waiting for $expected_nodes Ray node(s) to register (workers must be started with NODE_ROLE=worker, HEAD_ADDR=$HEAD_ADDR) ..."
    local waited=0 timeout="${RAY_WAIT_TIMEOUT:-600}" alive
    while :; do
        alive="$(python3 - <<'PY' 2>/dev/null || echo 0
import ray
ray.init(address="auto", logging_level="ERROR")
print(sum(1 for n in ray.nodes() if n.get("Alive")))
PY
)"
        alive="${alive:-0}"
        if [[ "$alive" -ge "$expected_nodes" ]]; then
            log "Ray cluster ready: $alive/$expected_nodes node(s) alive."
            break
        fi
        if [[ "$waited" -ge "$timeout" ]]; then
            die "Timed out after ${timeout}s waiting for Ray nodes ($alive/$expected_nodes alive). Check the worker container started, HEAD_ADDR is reachable from it, and the RoCE fabric is up."
        fi
        log "  ... $alive/$expected_nodes node(s) alive (waited ${waited}s)"
        sleep 5; waited=$(( waited + 5 ))
    done

    resolve_model_path
    maybe_build_oproj_overlay
    build_vllm_cmd
    log "Launching vLLM (head / rank 0):"
    log "  vllm ${VLLM_ARGS[*]}"
    exec vllm "${VLLM_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Ray worker: pre-stage weights (so vLLM on the head can place rank 1 here),
# join the head, and block.
# ---------------------------------------------------------------------------
run_worker() {
    [[ -n "${HEAD_ADDR:-}" ]] || die "NODE_ROLE=worker requires HEAD_ADDR (the head node's IP to join). e.g. -e HEAD_ADDR=<HEAD_IP>"

    # The worker rank also reads the weights locally — make sure they are here.
    resolve_model_path

    local self_ip="${WORKER_ADDR:-}"
    log "Joining Ray head at $HEAD_ADDR:$RAY_PORT as worker ..."
    ray start \
        --address="$HEAD_ADDR:$RAY_PORT" \
        ${self_ip:+--node-ip-address="$self_ip"} \
        --num-gpus=1 \
        --disable-usage-stats \
        --block
    # --block keeps the worker alive; vLLM on the head drives rank 1 here.
}

# ---------------------------------------------------------------------------
# Solo: single node, no Ray (for testing on one large GPU).
# ---------------------------------------------------------------------------
run_solo() {
    log "Solo mode: single-node, no Ray (TP_SIZE=$TP_SIZE)."
    resolve_model_path
    maybe_build_oproj_overlay
    build_vllm_cmd
    log "Launching vLLM (solo):"
    log "  vllm ${VLLM_ARGS[*]}"
    exec vllm "${VLLM_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# If the user passed an explicit command (docker run ... <cmd>), run it as-is.
# Otherwise dispatch by role.
# ---------------------------------------------------------------------------
if [[ "$#" -gt 0 ]]; then
    log "Explicit command given; exec'ing it verbatim: $*"
    exec "$@"
fi

case "$NODE_ROLE" in
    head)   run_head   ;;
    worker) run_worker ;;
    solo)   run_solo   ;;
esac
