#!/bin/bash
# Helper: run a bug's reproduce.sh on eval-gpu-0 via SSH
# Usage: bash run_on_gpu.sh M-033
set -euo pipefail

BUG="${1:?Usage: bash run_on_gpu.sh M-XXX}"
BENCH="/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/benchmark/bugs"
MEGATRON="/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/exp/frameworks/Megatron-LM"

# Write a temp script on shared filesystem
TMPSCRIPT="/volume/qscai/cqs/temp/run_${BUG}.sh"
mkdir -p /volume/qscai/cqs/temp
cat > "$TMPSCRIPT" << EOF
#!/bin/bash
source /etc/shinit_v2 2>/dev/null || true
export PATH="/opt/venv/bin:/usr/local/bin:\$PATH"
cd $BENCH/$BUG
MEGATRON_DIR=$MEGATRON bash reproduce.sh
EOF
chmod +x "$TMPSCRIPT"

ssh eval-gpu-0 "bash -l $TMPSCRIPT"
