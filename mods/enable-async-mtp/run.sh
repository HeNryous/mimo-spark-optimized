#!/bin/bash
# Enable async scheduling for MTP speculative decoding on the Ray executor.
#
# vLLM disables async scheduling for MTP spec-decode, but this is a conservative
# oversight, not a technical limitation:
#   1. The Ray distributed executor already implements async execution
#      (execute_model(non_block=True) -> FutureWrapper, max_concurrent_batches=2)
#      but never sets supports_async_scheduling()=True (inherits the base False).
#   2. vllm/config/vllm.py exempts "draft_model" from the async-disable gates but
#      forgets "mtp" — even though the hard-fail error message itself lists
#      "EAGLE/MTP/Draft Model/NGram" as supported.
# Async is lossless (identical tokens; only overlaps CPU scheduling with GPU exec).
# On a GPU-bound decode (~90% util) this recovers the CPU-scheduling gap:
# measured +6-8% tok/s across short and long context on 2x GB10 (TP=2, Ray).
# Requires --async-scheduling on the launch command.
set -euo pipefail
SITE="/usr/local/lib/python3.12/dist-packages"
echo "[enable-async-mtp] patching vLLM to allow async scheduling with MTP on Ray"
python3 - "$SITE" <<'PYEOF'
import ast, sys
SITE=sys.argv[1]
# 1) Ray executor: advertise async support (implementation already present)
f=SITE+"/vllm/v1/executor/ray_executor.py"; s=open(f).read()
if "supports_async_scheduling = classmethod(lambda" not in s:
    s=s.rstrip()+"\n\n# async impl present (non_block Future); enable the capability flag\nRayDistributedExecutor.supports_async_scheduling = classmethod(lambda cls: True)\n"
    ast.parse(s); open(f,"w").write(s); print("[enable-async-mtp] ray supports_async_scheduling -> True")
else:
    print("[enable-async-mtp] ray already patched")
# 2) vLLM config: exempt 'mtp' from both async-disable gates
f=SITE+"/vllm/config/vllm.py"; s=open(f).read()
a1=('                    self.speculative_config.method not in get_args(EagleModelTypes)\n'
    '                    and self.speculative_config.method not in get_args(NgramGPUTypes)\n'
    '                    and self.speculative_config.method != "draft_model"\n'
    '                ):')
b1=a1.replace('                ):',
              '                    and self.speculative_config.method != "mtp"\n                ):')
if 'method != "mtp"' not in s and a1 in s:
    s=s.replace(a1,b1,1); print("[enable-async-mtp] explicit-check exempts mtp")
a2=('                and self.speculative_config.method not in get_args(EagleModelTypes)\n'
    '                and self.speculative_config.method not in get_args(NgramGPUTypes)\n'
    '            ):')
b2=a2.replace('            ):',
              '                and self.speculative_config.method != "mtp"\n            ):')
if b2 not in s and a2 in s:
    s=s.replace(a2,b2,1); print("[enable-async-mtp] auto-check exempts mtp")
ast.parse(s); open(f,"w").write(s)
print("[enable-async-mtp] done")
PYEOF
