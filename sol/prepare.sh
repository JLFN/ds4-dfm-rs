#!/usr/bin/env bash
# Pin and fetch NVIDIA's Sol-H3 Spark pipeline, and put the one library it
# needs beyond Python into a venv.
#
# This subtree is a separate engine: it shares no code with this repository's
# Rust/CUDA path and nothing here is on a gate path. The script is idempotent
# and fetches third-party sources into sol/upstream, which is gitignored, so no
# upstream code enters this repository.
#
# What it does NOT do, because it needs the operator (see sol/README.md):
# accept the LTX-2.5 licence and log in, download weights, build the three
# upstream environments, write paths.json and the offline prompt cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# The upstream this tree was written against: NVlabs/Sana, branch sol-engine.
# Pinned to the commit the recon read (docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md).
UPSTREAM_URL="${SOL_UPSTREAM_URL:-https://github.com/NVlabs/Sana.git}"
UPSTREAM_BRANCH="${SOL_UPSTREAM_BRANCH:-sol-engine}"
UPSTREAM_COMMIT="${SOL_UPSTREAM_COMMIT:-6e4a440}"

# The package directory inside upstream, and the shared modules it imports.
PACKAGE="models/minimax_h3/Sol-H3-Spark"
CONE=("$PACKAGE" models/minimax_h3/Sol-H3 models/minimax_h3/GB200 \
      models/minimax_h3/super_acceleration techniques/sparse_backends)

# cuDNN, for Stage 1's BSA backend only. Pinned to what upstream's Spark
# pipeline pins and validated: the CUDA 13 build, which runs beside the
# toolkit already installed here without changing it. The current cuDNN release
# documents a newer minimum driver than this host runs, so this does not float.
CUDNN_VERSION="${SOL_CUDNN_VERSION:-9.20.0.48}"
CUDNN_FRONTEND_VERSION="${SOL_CUDNN_FRONTEND_VERSION:-1.27.0}"

UPSTREAM="$HERE/upstream"
VENV="$HERE/venvs/cudnn"
PYTHON="${SOL_PYTHON:-python3}"

say() { printf '%s\n' "$*"; }

fetch_upstream() {
    if [ -d "$UPSTREAM/.git" ]; then
        say "upstream: already present at $UPSTREAM"
    else
        say "upstream: fetching $UPSTREAM_URL ($UPSTREAM_BRANCH)"
        git clone --filter=blob:none --sparse --branch "$UPSTREAM_BRANCH" \
            "$UPSTREAM_URL" "$UPSTREAM"
    fi
    # A cheap-nothing checkout: one package plus the shared modules it imports.
    git -C "$UPSTREAM" sparse-checkout set "${CONE[@]}"
    if git -C "$UPSTREAM" cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null; then
        git -C "$UPSTREAM" checkout --quiet "$UPSTREAM_COMMIT"
        say "upstream: at ${UPSTREAM_COMMIT} ($(git -C "$UPSTREAM" log -1 --format=%s))"
    else
        say "upstream: commit ${UPSTREAM_COMMIT} not in this checkout; staying on $(git -C "$UPSTREAM" rev-parse --short HEAD)"
        say "upstream: set SOL_UPSTREAM_COMMIT to pin, or run git -C $UPSTREAM fetch first"
    fi
    if [ ! -f "$UPSTREAM/$PACKAGE/README.md" ]; then
        say "upstream: $PACKAGE is missing; the sparse cone did not materialize"
        exit 1
    fi
}

prepare_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        say "venv: creating $VENV"
        "$PYTHON" -m venv "$VENV"
    fi
    say "venv: pinning cuDNN $CUDNN_VERSION and frontend $CUDNN_FRONTEND_VERSION"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet \
        "nvidia-cudnn-cu13==$CUDNN_VERSION" \
        "nvidia-cudnn-frontend==$CUDNN_FRONTEND_VERSION"
    say "venv: installed under $(dirname "$("$VENV/bin/python" -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')")"
}

check_host() {
    if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
        say "host: no nvcc found; this pipeline targets one GB10 and is not portable"
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        say "host: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        say "host: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    fi
    say "host: free memory"
    free -g | head -2
}

main() {
    check_host
    fetch_upstream
    prepare_venv
    cat <<EOF

Prepared. What still needs the operator (sol/README.md has the detail):

  1. Accept the LTX-2.5 access terms and: hf auth login
  2. Download the weights:
       $VENV/bin/python $UPSTREAM/$PACKAGE/download_checkpoints.py --plan
       $VENV/bin/python $UPSTREAM/$PACKAGE/download_checkpoints.py \\
         --output-dir $HERE/checkpoints --include-offline
  3. Build the three upstream environments: $UPSTREAM/$PACKAGE/docs/setup.md
  4. Write the path manifest and build the offline prompt cache:
       $VENV/bin/python $UPSTREAM/$PACKAGE/prepare.py --plan

Then run one request with: $HERE/run.sh "<prompt>" <seed> outputs/<name>

Note: Stage 1 wants cuDNN from its own environment; point that environment's
PYTHONPATH at $VENV/lib/python3*/site-packages if it does not see the pinned
cuDNN, and confirm the version it reports before trusting a timing.
EOF
}

main "$@"
