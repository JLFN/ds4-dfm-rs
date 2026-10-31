#!/bin/bash
# Prompt -> 1344x768 clip with the content the draft drew, in one command.
#
# Stage 1 renders the H3 draft latent and its soundtrack; no H3 video is
# decoded (the reference's own route). Stage 2 upscales the latent 2x in H3
# space and decodes it with the H3 video VAE. There is no Latent Transfer
# Adapter, no LTX refiner, no LTX VAE and no Gemma: the refiner starts from
# sigma 0.909375 and rewrites 90.9 percent of the frame, which is why every
# refined clip came back as a different animal.
#
# Usage:
#   tools/make-h3-clip.sh -p "prompt" [-o out.mp4] [-d model_dir]
#     [--width N --height N --frames N --steps N --seed N] [--direct]
#
#   --direct renders at the output canvas itself (no draft, no upscale); it is
#   slower and sharper, and it needs the non-distilled checkpoint.
set -u

ROOT=${H3_ROOT:-$HOME/ds4-sol-h3}
A=${H3_ARTIFACTS:-/mnt/h3-artifacts}
MODEL=$A/MiniMax-H3-FastH3-4step
NVFP4=$A/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
UPSCALER=$A/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
H3_VAE=$A/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source
PROMPT=""
OUT=$ROOT/outputs/clip.mp4
WIDTH=672
HEIGHT=384
FRAMES=124
STEPS=4
STEPS_SET=0
SEED=42
DIRECT=0

while [ $# -gt 0 ]; do
  case "$1" in
    -p) PROMPT=$2; shift 2 ;;
    -o) OUT=$2; shift 2 ;;
    -d) MODEL=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --steps) STEPS=$2; STEPS_SET=1; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --direct) DIRECT=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PROMPT" ]; then
  echo "missing -p prompt" >&2
  exit 2
fi

# The two-stage route's stage 2 is the release's own geometry: its upscale
# target and adapter shape are 37 latent frames of 48x84, which is 124 frames
# at 672x384. Refuse another length before stage 1 spends its load, instead of
# leaving the saved latent to fail the check in h3-sol-rs.
if [ "$DIRECT" = "0" ] && [ "$FRAMES" != "124" ]; then
  echo "the two-stage route renders 124 frames; --frames $FRAMES needs --direct" >&2
  exit 2
fi

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$OUT")" scratch

export H3_DIT_SOL=0.9
# The decoder's arms: the campaign's, unless the operator states its own. `bf16i`
# is the faster one (48.5 s against 60.5 at 1344x768) and the decode line in the
# log names whichever ran.
: "${H3_VAE_ATTENTION:=tensor}"
: "${H3_VAE_GEMM:=bf16}"
export H3_VAE_ATTENTION H3_VAE_GEMM

guard() {
  local log=$1
  shift
  python3 tools/host_memory_guard.py --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 3600 \
    --log "scratch/$log.memory.jsonl" -- "$@"
}

started=$(date +%s.%N)

if [ "$DIRECT" = "1" ]; then
  # One pass at the output canvas: nothing is upscaled afterwards.
  direct_steps=$STEPS
  if [ "$STEPS_SET" = "0" ]; then
    direct_steps=20
  fi

  guard clip-direct "$ROOT/h3-rs" -d "$A/MiniMax-H3" --text-nvfp4 "$NVFP4" \
    -p "$PROMPT" --width 1344 --height 768 --frames "$FRAMES" \
    --steps "$direct_steps" --seed "$SEED" -o "$OUT" || exit 1
else
  LAT=scratch/clip.bin
  PCM=scratch/clip.pcm

  guard clip-stage1 "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
    -p "$PROMPT" --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
    --steps "$STEPS" --seed "$SEED" \
    --save-latent "$ROOT/$LAT" --save-audio "$ROOT/$PCM" --latent-only || exit 1

  guard clip-stage2 "$ROOT/h3-sol-rs" --latent "$ROOT/$LAT" --audio "$ROOT/$PCM" \
    --upscaler "$UPSCALER" --h3-vae "$H3_VAE" --h3-video upscaled -o "$OUT" || exit 1
fi

ended=$(date +%s.%N)

awk -v a="$started" -v b="$ended" 'BEGIN {printf "WALL total=%.3f\n", b-a}'
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,sample_rate,channels \
  -of default=noprint_wrappers=1 "$OUT"
ffmpeg -v error -i "$OUT" -f null - && echo DECODE_OK
sha256sum "$OUT"
echo "CLIP $OUT"
