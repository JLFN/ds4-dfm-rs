#!/bin/bash
# One-time setup of the clip-scoring venv that tools/h3-clip-score.py needs (DGX Spark, aarch64, GB10, CUDA 13).
#
#   tools/h3-clip-score-venv.sh [VENV_DIR]        # ~10-15 min here, ~2 GB of wheels, ~0.9 GB of weights on first use
#
# The versions are the reference campaign's own (`harness/eval_setup.sh` in the Rapid-H3 tree, validated
# 2026-10-24 on this host), trimmed to the criteria this gate uses: torch + pyiqa (MUSIQ, CLIP-IQA) +
# scikit-image (wavelet noise) + the frame decoders. The harness's other criteria (identity, voice,
# lip sync, dialogue, reference image) need insightface/whisper/demucs/speechbrain/SyncNet and are not
# installed here, because this port has no reference images or shot scripts to score against.
#
# DOVER is fetched with the campaign's pin and weights, into $ROOT/DOVER (the --dover-dir default).
# The harness itself is not fetched: point H3_EVAL_HARNESS at a copy of the reference tree's
# harness/eval_clips.py (tools/h3-clip-score.py prints its sha256 on every run, and the copy must match).
set -euo pipefail
ROOT=${H3_EVAL_ROOT:-$HOME/ds4-test}
VENV=${1:-$ROOT/venv_eval}
IDX=https://download.pytorch.org/whl/cu130
mkdir -p "$ROOT"

[ -x "$VENV/bin/python" ] || ${UV:-uv} venv "$VENV" --python 3.12
PIP="${UV:-uv} pip install --python $VENV/bin/python --index-strategy unsafe-best-match"

# --index-strategy unsafe-best-match: without it uv takes the first index that carries a name, and the
# torch index carries an old setuptools, so the pinned 80.10.2 below is reported unsatisfiable.
$PIP --index-url $IDX torch==2.14.1+cu130 torchvision==0.29.1+cu130
$PIP --extra-index-url $IDX \
    numpy==2.5.2 scipy==1.18.1 scikit-image==0.26.0 PyWavelets==1.10.0 \
    opencv-python-headless==5.0.0.93 av==19.0.2 pyiqa==0.1.16 PyYAML==6.0.3 einops==0.8.2 \
    "setuptools==80.10.2"

# DOVER (VQAssessment/DOVER) + DOVER.pth, at the campaign's pin
if [ ! -d "$ROOT/DOVER" ]; then
    git clone -q https://github.com/VQAssessment/DOVER.git "$ROOT/DOVER"
    git -C "$ROOT/DOVER" checkout -q f1ddc96215bc7fbcf8f315c65d47905f339c3419
fi
mkdir -p "$ROOT/DOVER/pretrained_weights"
[ -s "$ROOT/DOVER/pretrained_weights/DOVER.pth" ] || \
    wget -q https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth \
        -O "$ROOT/DOVER/pretrained_weights/DOVER.pth"

"$VENV/bin/python" - <<PY
import torch, pyiqa, skimage, cv2, av
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("pyiqa", pyiqa.__version__, "skimage", skimage.__version__, "cv2", cv2.__version__, "av", av.__version__)
for name in ("musiq", "clipiqa"):  # first use downloads the weights into the harness's cache
    m = pyiqa.create_metric(name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(name, "ready:", type(m).__name__)
PY
echo "h3-clip-score venv ready: $VENV"
