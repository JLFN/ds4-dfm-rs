#!/usr/bin/env bash
# One Sol-H3 request, through upstream's own entry point in sol/upstream.
#
#   ./sol/run.sh "<prompt>" [seed] [output-dir]
#
# This wrapper exists so the invocation is recorded in one place: the engine's
# numbers and this pipeline's numbers are different contracts and must never be
# blended. It does not touch the engine, and it does not patch upstream.
#
# Run one GPU job at a time. Upstream's timing counts the first formal request
# and excludes only its explicit warmup, so a single run is not a hot figure.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="$HERE/upstream/models/minimax_h3/Sol-H3-Spark"
VENV="$HERE/venvs/cudnn"

PROMPT="${1:-}"
SEED="${2:-42}"
OUTDIR="${3:-$HERE/outputs/run-$(date +%Y%m%d-%H%M%S)}"

if [ -z "$PROMPT" ]; then
    sed -n '3,8p' "$0"
    exit 2
fi
if [ ! -f "$PACKAGE/infer.py" ]; then
    echo "sol: $PACKAGE/infer.py is missing; run ./sol/prepare.sh first" >&2
    exit 1
fi
if [ ! -f "$HERE/paths.json" ]; then
    echo "sol: $HERE/paths.json is missing; see sol/README.md step 4 (prepare.py)" >&2
    exit 1
fi
if [ -e "$OUTDIR" ]; then
    echo "sol: refusing to reuse an existing output directory: $OUTDIR" >&2
    exit 1
fi

# Upstream requires its interpreter selection: Stage1 and Stage2 are separate
# environments and the Qwen stage runs through its container wrapper. Point
# SOL_STAGE1_PYTHON / SOL_STAGE2_PYTHON / SOL_QWEN_PYTHON at them, or accept
# the venv this tree prepared, which carries cuDNN only.
PYTHON="${SOL_STAGE1_PYTHON:-$VENV/bin/python}"

echo "sol: prompt  ${PROMPT:0:60}..."
echo "sol: seed    $SEED"
echo "sol: output  $OUTDIR"
echo "sol: python  $PYTHON"
echo "sol: upstream $(git -C "$HERE/upstream" rev-parse --short HEAD 2>/dev/null || echo unknown)"

cd "$PACKAGE"
exec "$PYTHON" infer.py \
    --paths "$HERE/paths.json" \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --output-dir "$OUTDIR"
