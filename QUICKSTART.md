# Quick start — pull & run (MiMo-V2.5-NVFP4 on 2x DGX Spark)

Pull the image, set **node role + peer IP**, run. The head node forms a Ray
cluster, waits for the worker, then serves vLLM with the optimized **Config C**
(4-bit nvfp4 KV-cache + WMMA decode + `NCCL_CROSS_NIC` multi-rail + cudagraph
boot-limit + MTP). **The model auto-downloads on first start.**

Replace the placeholders:

| Placeholder    | Meaning                                                |
|----------------|--------------------------------------------------------|
| `<OWNER>`      | your registry namespace, e.g. a GHCR user/org          |
| `<HEAD_IP>`    | IP of **node 0** (the Ray head / rank 0 / the endpoint)|
| `<WORKER_IP>`  | IP of **node 1** (the Ray worker / rank 1)             |

> **Hardware:** this stack is **sm_121 specific** — 2x NVIDIA DGX Spark (GB10,
> ARM aarch64), `TP=2` over RoCE. It will not run elsewhere without rebuilding
> the base image + kernels. See [`docs/MULTI_NODE_SETUP.md`](docs/MULTI_NODE_SETUP.md).

---

## 1. Persistent model cache (do this first)

The NVFP4 weights are **~171 GB** and **both nodes need them** (each rank reads
its own copy). Mount a persistent volume at the HF cache so they download **once**
per node instead of on every container start:

```bash
# Run on BOTH nodes (creates a named docker volume for the HF cache):
docker volume create hf-cache
```

(Or bind-mount a host directory with `-v /path/on/host:/root/.cache/huggingface`.)

---

## 2. Start the worker (node 1) first

```bash
docker run -d --name mimo-worker \
  --gpus all --network host --ipc=host \
  -v hf-cache:/root/.cache/huggingface \
  -e NODE_ROLE=worker \
  -e HEAD_ADDR=<HEAD_IP> \
  ghcr.io/<OWNER>/mimo-spark-optimized:latest
```

The worker pre-stages the weights (first start: long download), joins the head's
Ray cluster, and blocks. It does **not** serve HTTP — the head does.

## 3. Start the head (node 0)

```bash
docker run -d --name mimo-head \
  --gpus all --network host --ipc=host \
  -v hf-cache:/root/.cache/huggingface \
  -e NODE_ROLE=head \
  -e HEAD_ADDR=<HEAD_IP> \
  -p 8000:8000 \
  ghcr.io/<OWNER>/mimo-spark-optimized:latest
```

The head starts the Ray head, **waits for the worker to register** (`TP_SIZE=2`
=> 2 nodes), auto-downloads the model if needed, then runs `vllm serve` with the
Config-C arguments across both ranks.

> **First start takes a while:** ~171 GB download per node (skip-able if the
> cache volume is pre-populated) **plus** a one-time kernel JIT + cudagraph
> capture (the `cudagraph_capture_sizes=[1,2,4,8]` boot-limit keeps this to a
> few minutes instead of ~28). Subsequent starts reuse the volume and are fast.

Follow progress with `docker logs -f mimo-head`. It is ready when the log shows
the vLLM server started and the KV-cache size (~400–500K tokens).

---

## 4. Test the endpoint

```bash
# Model is serving:
curl -s http://<HEAD_IP>:8000/v1/models

# A short chat completion:
curl -s http://<HEAD_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "MiMo-V2.5-NVFP4",
        "messages": [{"role": "user", "content": "In one sentence, what is RoCE?"}],
        "max_tokens": 64
      }'
```

---

## Single-node test (one large GPU, no Ray)

For trying the stack on a single GPU big enough to hold the model, run **solo**
mode (no Ray, one process). Note the real target is 2x GB10 — on one GB10 the
model does **not** fit:

```bash
docker run --rm \
  --gpus all --network host --ipc=host \
  -v hf-cache:/root/.cache/huggingface \
  -e NODE_ROLE=solo \
  -e TP_SIZE=1 \
  -p 8000:8000 \
  ghcr.io/<OWNER>/mimo-spark-optimized:latest
```

(`NODE_ROLE` unset together with `TP_SIZE=1` also defaults to solo.)

---

## Tunables (optional `-e` overrides)

| Env var                  | Default                      | Notes                                              |
|--------------------------|------------------------------|----------------------------------------------------|
| `MODEL`                  | `lukealonso/MiMo-V2.5-NVFP4` | HF id or local path                                |
| `TP_SIZE`                | `2`                          | tensor-parallel size (=node count)                 |
| `GPU_MEM_UTIL`           | `0.87`                       | Config-C util ceiling (see docs; do not raise blindly) |
| `MAX_MODEL_LEN`          | `98304`                      | 96K context                                        |
| `MAX_NUM_SEQS`           | `6`                          | raise (~24) for the aggregate-throughput sweet spot|
| `NCCL_IB_HCA`            | auto-detect                  | override the RoCE HCA list if detection is wrong   |
| `ENABLE_OPROJ_MXFP8`     | off                          | `1` = build + serve the `o_proj`→MXFP8 KV overlay (+~8.5% KV) |
| `EXTRA_VLLM_ARGS`        | empty                        | appended verbatim to `vllm serve`                  |

The `o_proj`→MXFP8 overlay (`ENABLE_OPROJ_MXFP8=1`) is generated on the head with
`tools/quantize_oproj_mxfp8.py` on first start and cached in the HF volume; it is
**off by default** to keep pull & run simple.

---

## Troubleshooting

- **Head times out waiting for Ray nodes** — confirm the worker container is up
  (`docker logs mimo-worker`), `HEAD_ADDR` is reachable from the worker, and the
  RoCE fabric is healthy. See [`docs/MULTI_NODE_SETUP.md`](docs/MULTI_NODE_SETUP.md).
- **NCCL fell back to `[Socket]`** — set `-e NCCL_IB_HCA=...` and
  `-e NCCL_SOCKET_IFNAME=...` explicitly; check the NCCL `INFO` log shows IB/RoCE.
- **"No GPU visible"** — add `--gpus all`.
- **Box freezes / memory pressure** — GB10 is unified memory and does not OOM;
  keep `GPU_MEM_UTIL` at the documented `0.87`, keep a small swap, and read the
  unified-memory section of [`docs/MULTI_NODE_SETUP.md`](docs/MULTI_NODE_SETUP.md).
