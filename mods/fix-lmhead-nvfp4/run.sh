#!/bin/bash
set -euo pipefail
# fix-lmhead-nvfp4
# Route a ParallelLMHead whose quant_algo is W4A16_NVFP4 (or NVFP4) through
# ModelOptNvFp4W4A16LinearMethod (weight-only 4-bit FP4 Marlin, bf16 activations).
# vLLM's ModelOptMixedPrecisionConfig.get_quant_method only routes LinearBase /
# RoutedExperts; a ParallelLMHead (VocabParallelEmbedding subclass) otherwise
# falls back to UnquantizedEmbeddingMethod and crashes on the nvfp4 scale params.
# VocabParallelEmbedding.__init__ DOES call self.quant_method.create_weights(...)
# and its weight_loader shards weight/weight_scale on the vocab (output) dim and
# copies the scalar weight_scale_2 -> the nvfp4 W4A16 triple loads cleanly.
# Gate validated standalone on sm_121: Marlin W4A16 rel-err 0.008, ~5x faster than
# bf16 at decode batch (lmhead-nvfp4-microtest.py / -speedtest.py).
SITE="/usr/local/lib/python3.12/dist-packages"
F="$SITE/vllm/model_executor/layers/quantization/modelopt.py"
echo "[fix-lmhead-nvfp4] patching $F"
python3 - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py")
t = p.read_text()
MARK = "fix-lmhead-nvfp4 mod"
if MARK in t:
    print("[fix-lmhead-nvfp4] already applied — skip"); raise SystemExit(0)

anchor = ("        quant_algo = self._resolve_quant_algo(prefix)\n"
          "\n"
          "        if isinstance(layer, LinearBase):\n"
          "            if quant_algo == \"FP8\":")
assert t.count(anchor) == 1, f"anchor not unique/found: {t.count(anchor)}"

inject = (
"        quant_algo = self._resolve_quant_algo(prefix)\n"
"\n"
"        # fix-lmhead-nvfp4 mod: a ParallelLMHead (VocabParallelEmbedding\n"
"        # subclass) is neither LinearBase nor RoutedExperts, so without this\n"
"        # branch it falls back to UnquantizedEmbeddingMethod and crashes on the\n"
"        # nvfp4 scale params. Route a W4A16_NVFP4 lm_head through the weight-only\n"
"        # FP4 Marlin linear method; the logits path already calls\n"
"        # lm_head.quant_method.apply(...). Marlin reads 4-bit weights (true BW\n"
"        # win, NOT dequant-to-bf16 emulation). Validated on sm_121.\n"
"        from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
"            ParallelLMHead as _ParallelLMHead,\n"
"        )\n"
"        if isinstance(layer, _ParallelLMHead):\n"
"            # Omni wrapper gives prefix 'language_model.lm_head' but the config\n"
"            # key stays 'lm_head' (mapper does not prefix it) -> direct/prefix\n"
"            # resolve misses. Fall back to matching the leaf component.\n"
"            if quant_algo is None:\n"
"                _leaf = prefix.rsplit(\".\", 1)[-1]\n"
"                for _k, _info in self.quantized_layers.items():\n"
"                    if _k == _leaf or _k.endswith(\".\" + _leaf):\n"
"                        quant_algo = _info[\"quant_algo\"].upper()\n"
"                        break\n"
"            if quant_algo in (\"W4A16_NVFP4\", \"NVFP4\"):\n"
"                _base = getattr(self, \"nvfp4_config\", None)\n"
"                _gs = getattr(_base, \"group_size\", 16) if _base is not None else 16\n"
"                _kv = getattr(_base, \"kv_cache_quant_algo\", None) if _base is not None else None\n"
"                _w4a16 = ModelOptNvFp4Config(\n"
"                    quant_method=\"W4A16_NVFP4\",\n"
"                    is_checkpoint_nvfp4_serialized=True,\n"
"                    kv_cache_quant_algo=_kv,\n"
"                    exclude_modules=[],\n"
"                    group_size=_gs,\n"
"                )\n"
"                return ModelOptNvFp4W4A16LinearMethod(_w4a16)\n"
"            return None\n"
"\n"
"        if isinstance(layer, LinearBase):\n"
"            if quant_algo == \"FP8\":"
)
t2 = t.replace(anchor, inject, 1)
assert t2 != t and MARK in t2
p.write_text(t2)
print("[fix-lmhead-nvfp4] injected ParallelLMHead W4A16_NVFP4 branch")
PY
python3 -c "import ast; ast.parse(open('$F').read()); print('[fix-lmhead-nvfp4] ast OK')"
python3 -m py_compile "$F" && echo "[fix-lmhead-nvfp4] py_compile OK"

# --- patch 2: VocabParallelEmbedding must expose params_dtype (Marlin fp4
#     prepare_fp4_layer_for_marlin reads layer.params_dtype; ParallelLMHead does
#     not set it, unlike LinearBase). ---
VPE="$SITE/vllm/model_executor/layers/vocab_parallel_embedding.py"
python3 - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py")
t = p.read_text()
if "LMHEAD_NVFP4_PARAMS_DTYPE" in t:
    print("[fix-lmhead-nvfp4] params_dtype already set"); raise SystemExit(0)
anchor = "        self.quant_method.create_weights(\n            self,"
assert t.count(anchor) == 1, f"vpe anchor count {t.count(anchor)}"
inj = ("        # LMHEAD_NVFP4_PARAMS_DTYPE: Marlin fp4 prep reads layer.params_dtype\n"
       "        self.params_dtype = params_dtype\n" + anchor)
p.write_text(t.replace(anchor, inj, 1))
print("[fix-lmhead-nvfp4] set VocabParallelEmbedding.params_dtype")
PY
python3 -m py_compile "$VPE" && echo "[fix-lmhead-nvfp4] vpe py_compile OK"

# --- patch 3: MTP drafter lm_head must also get quant_config, else it builds an
#     unquantized [vocab,hidden] head and the packed nvfp4 lm_head.weight
#     [vocab,hidden/2] fails to load into it (mimo_v2_mtp.py). ---
MTP="$SITE/vllm/model_executor/models/mimo_v2_mtp.py"
python3 - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py")
t = p.read_text()
if "LMHEAD_NVFP4_MTP_QUANT" in t:
    print("[fix-lmhead-nvfp4] mtp quant_config already set"); raise SystemExit(0)
anchor = ('        self.lm_head = ParallelLMHead(\n'
          '            self.config.vocab_size,\n'
          '            self.config.hidden_size,\n'
          '            prefix=maybe_prefix(prefix, "lm_head"),\n'
          '        )')
assert t.count(anchor) == 1, f"mtp anchor count {t.count(anchor)}"
inj = ('        # LMHEAD_NVFP4_MTP_QUANT: route drafter lm_head through quant_config too\n'
       '        self.lm_head = ParallelLMHead(\n'
       '            self.config.vocab_size,\n'
       '            self.config.hidden_size,\n'
       '            quant_config=vllm_config.quant_config,\n'
       '            prefix=maybe_prefix(prefix, "lm_head"),\n'
       '        )')
p.write_text(t.replace(anchor, inj, 1))
print("[fix-lmhead-nvfp4] MTP drafter lm_head now uses quant_config")
PY
python3 -m py_compile "$MTP" && echo "[fix-lmhead-nvfp4] mtp py_compile OK"
