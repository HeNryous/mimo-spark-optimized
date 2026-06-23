# TK-decode deploy (staged 2026-06-23) — NOT YET DEPLOYED

Validated drop-in: split-K + MTP-fusion + pipelined-dequant + boundary-masking flash-decode.
**2.49–2.87× over the WMMA kernel** on the MTP (q_len=3) decode path. Correct (rel ~2.3e-3),
cudagraph-safe (static scratch, NSPLIT=f(num_seqs) static per capture), full gate parity
(SWA/softcap/prefill/ragged → fall back).

## Staged artifacts (live files untouched)
- `mods/nvfp4-kv-diffkv/tk-decode/tk_decode.py`   — the module (try_wmma_decode drop-in)
- `mods/nvfp4-kv-diffkv/tk-decode/include`, `/prototype` — ThunderKittens headers (2MB)

## Deploy steps (require MiMo restart)
1. **run.sh** — append after the existing wmma_decode.py copy:
   ```sh
   # ThunderKittens decode kernel (2.6x over WMMA on MTP path)
   mkdir -p /opt/tkinc
   cp -r "$HERE/tk-decode/include"   /opt/tkinc/include
   cp -r "$HERE/tk-decode/prototype" /opt/tkinc/prototype
   cp "$HERE/tk-decode/tk_decode.py" "$SITE/vllm/v1/attention/ops/tk_decode.py"
   ```
2. **wmma_decode.py** — at the very top of `try_wmma_decode(...)`, add env-gated delegation:
   ```python
   if os.environ.get("VLLM_TK_DECODE","0") == "1":
       try:
           from vllm.v1.attention.ops import tk_decode as _tk
           r = _tk.try_wmma_decode(q,k_cache,out,seqused_k,block_table,softmax_scale,
                   num_kv_heads,head_size_qk,head_size_v,block_size,sinks,softcap,
                   window_left,cu_seqlens_q,max_seqlen_q,force)
           if r: return True       # TK handled it; else fall through to WMMA below
       except Exception:
           pass                    # any TK failure -> safe WMMA fallback
   ```
3. **recipe env** (mimo-v2.5-nvfp4.yaml): set `TK_INC: "/opt/tkinc"` and `VLLM_TK_DECODE: "1"`.
   (Keep VLLM_WMMA_DECODE=1 — TK falls back to WMMA on any unhandled shape.)
4. Restart MiMo. **Post-boot verify:**
   - `curl :8000/v1/models` healthy; boot-log shows cudagraph active (enforce_eager=False).
   - Smoke a long-context + an MTP gen for coherence (quality must match: rel ~2.3e-3 = nvfp4 noise).
   - `/tmp/wmma_trace.log` (WMMA path) should be quiet for full-attn decode if TK is taking it.
5. **Rollback:** set `VLLM_TK_DECODE=0` (instant, reverts to WMMA) or restart. No file changes needed to roll back.

## Risk notes
- First call JIT-compiles tk_decode (~1-2 min) → pre-compile at startup like wmma (`_compile()` in the
  diffkv backend startup hook) to avoid a stall on first decode.
- NSPLIT rule = `max(32, min(256, 256//num_seqs))`; scratch fixed [256, max_batch*qlen*NQH] (~once).
- Boundary masking validated for arbitrary L; ragged/non-uniform q_len falls back to WMMA (safe).
- TK uses sm_121a; same arch as prod. Headers are header-only (no extra runtime deps).

## Measured (standalone, in-container, flashinfer ref)
| regime | WMMA µs | TK µs | speedup |
|---|---|---|---|
| B1 L8192 | 1242 | 498 | 2.49× |
| B1 L131072 | 17668 | 6164 | 2.87× |
| B8 L4096 | 4367 | 1572 | 2.78× |
| B8 L32768 | 46574 | 17990 | 2.59× |
| B16 L2048 | 6052 | 2294 | 2.64× |

End-to-end tok/s gain is smaller (Amdahl: attention is part of the decode step) and largest for
MTP + long-context + multi-stream. Measure post-deploy with VLLM_TK_DECODE toggle for clean A/B.
