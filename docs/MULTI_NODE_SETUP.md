# Multi-node setup: MiMo-V2.5-NVFP4 on 2x DGX Spark (TP=2)

MiMo-V2.5-NVFP4 (~310B MoE, ~151 GB of NVFP4 weights) does not fit on a single
GB10 (128 GB unified memory). This guide runs it tensor-parallel across **two**
DGX Spark nodes over their RoCE interconnect, with vLLM driving the two ranks via
Ray.

Throughout, replace the placeholders with your own values:

| Placeholder    | Meaning                                  |
|----------------|------------------------------------------|
| `<NODE0_IP>`   | IP of node 0 (the Ray head / rank 0)     |
| `<NODE1_IP>`   | IP of node 1 (the Ray worker / rank 1)   |
| `node0`        | hostname of node 0                       |
| `node1`        | hostname of node 1                       |
| `<RoCE_IFACE>` | RDMA NIC name(s), e.g. `rocep1s0` etc.   |
| `<HCA_LIST>`   | NCCL HCA list, e.g. `rocep1s0,rocep2s0`  |

> Hardware note: this stack is **sm_121 specific** (GB10, `TORCH_CUDA_ARCH_LIST=12.1a`,
> ARM aarch64). It will not run on other GPUs without rebuilding the base image and
> the JIT-compiled CUDA/WMMA kernel for your architecture.

---

## 1. Network / fabric

The two GB10s are connected with two RoCE (RDMA-over-Converged-Ethernet) NICs.
The single biggest single-stream speed lever in this stack is using **both** NICs
as a multi-rail link (`NCCL_CROSS_NIC=1`, +42% single-stream on TP=2 — single-stream
decode here is comm/latency-bound, not weight-bandwidth-bound).

Recommended fabric config (apply on **both** nodes):

- **MTU 9000 (jumbo frames)** on the RoCE interfaces — large all-reduce buffers
  benefit. Verify with `ip link show <RoCE_IFACE>`.
- **RoCE / PFC** enabled on the NICs and the switch (or a direct cable) so RDMA
  traffic is lossless.
- Confirm RDMA is healthy with `ibstat` / `ib_write_bw` between the two nodes
  before launching vLLM.

NCCL environment (set identically on both nodes — also in the example recipe):

```bash
export NCCL_CROSS_NIC=1                 # multi-rail across both RoCE NICs (the hammer)
export NCCL_IB_HCA=<HCA_LIST>           # restrict NCCL to the RDMA NICs, e.g. rocep1s0,rocep2s0
export NCCL_SOCKET_IFNAME=<RoCE_IFACE>  # control-plane interface
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_BUFFSIZE=8388608
```

If NCCL falls back to TCP sockets instead of RDMA you will lose most of the
benefit — check the NCCL `INFO` log at startup shows the IB/RoCE transport on
the inter-node ring, not `[Socket]`.

---

## 2. Ray cluster (cross-node)

vLLM uses Ray as the distributed executor (`--distributed-executor-backend ray`).
Start a 2-node Ray cluster, then launch vLLM **once** on the head node — it will
place rank 0 locally and rank 1 on the worker.

On **node 0** (head):

```bash
ray start --head \
    --node-ip-address=<NODE0_IP> \
    --port=6379 \
    --num-gpus=1
```

On **node 1** (worker):

```bash
ray start \
    --address=<NODE0_IP>:6379 \
    --node-ip-address=<NODE1_IP> \
    --num-gpus=1
```

Verify both nodes joined (`ray status` should show 2 GPUs). Then launch the
optimized container on the head node with the recipe `command` (see
`recipes/mimo-v2.5-nvfp4.example.yaml`), using `--tensor-parallel-size 2` and
`--distributed-executor-backend ray`.

> Keep `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` and
> `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0` on this >=100 GB model — the V2
> executor / DAG-overlap path costs extra unified memory and can push the box
> into a memory-pressure freeze.

---

## 3. Unified-memory / freeze safety

GB10 is a **unified-memory** device (CPU + GPU share the 128 GB). The CUDA
allocator does **not** raise OOM — it requests endlessly, fills RAM, then swap,
then the box can hard-freeze (no OOM-killer fires). Practical guidance:

- The NVFP4 model rests at only ~2.4 GiB free at `gpu_memory_utilization=0.87`.
  Do **not** raise util blindly; `util < 0.86` yields *negative* KV (model +
  activations exceed the reservation).
- Keep a small swap (e.g. 16 GiB, `swappiness=1`) as a soft buffer.
- Run a lightweight watchdog that kills the container before a true freeze, e.g.
  when `SwapFree` drops below ~1.2 GiB (PSI `full avg10` is a useful early
  *indicator* but spikes legitimately during cudagraph capture, so gate the kill
  on swap exhaustion, not PSI alone).
- If a container stop leaves device residue (e.g. one rank shows free memory that
  the host does not), reload the UVM driver on both nodes
  (`rmmod nvidia_uvm; modprobe nvidia_uvm`) rather than lowering util.

---

## 4. Boot time

The first boot JIT-compiles kernels and captures cudagraphs. Limiting
`cudagraph_capture_sizes` to a small set (e.g. `[1,2,4,8]`) cut boot from
~28 min to ~4 min (~7x) and applies to every restart. This is set in the example
recipe via `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'`.

---

## 5. Sanity checks after launch

```bash
# Model is serving:
curl -s http://<NODE0_IP>:8000/v1/models

# A short completion (coherence):
curl -s http://<NODE0_IP>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-NVFP4","prompt":"17*23=","max_tokens":16}'
```

Confirm in the startup log: NCCL chose the RoCE/IB transport, the KV-cache size
("GPU KV cache size") is in the expected range (~400–500K tokens for the 4-bit
nvfp4 KV config), and FULL cudagraphs captured without
`cudaErrorStreamCaptureUnsupported`.
