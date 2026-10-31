#!/bin/bash
# Does the resident product route render the same clip as the two-process one?
#
# One request, three renders at one seed:
#
#   1. the same request twice in one process (`h3-rs --jobs`): the second must
#      hit the prompt encoder, the DiT and the upscaler, and must print the
#      first one's frames hash;
#   2. the two-process route this unit replaces: `h3-rs --latent-only`, then
#      `h3-sol-rs --h3-video upscaled --hash`.
#
# PASS needs all of it: job 2's text, DiT and upscaler loads reported as hits,
# job 2's frames hash equal to job 1's, and both equal to the two-process run's.
#
# Artifacts come from the environment, never from a path here:
#   H3_ARTIFACTS    the directory holding the checkpoints (required)
#   H3_ROOT         the repository checkout (default: the current directory)
#   H3_VAE_GEMM, H3_VAE_ATTENTION
#                   the decoder arms, when the operator states them; every run
#                   inherits them, so the comparison stays at one arm
#
# Usage: tools/h3-jobs-accept.sh [-p PROMPT] [--seed N] [--steps N]
#          [--width N --height N --frames N]
set -u

ROOT=${H3_ROOT:-$PWD}
A=${H3_ARTIFACTS:?H3_ARTIFACTS must name the checkpoint directory}
MODEL=$A/MiniMax-H3-FastH3-4step
NVFP4=$A/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
UPSCALER=$A/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
VAE=$MODEL/FL2VA/video_vae/source

PROMPT="A red fox trots through fresh snow at dawn, side profile, one tail, eye level."
SEED=42
STEPS=4
WIDTH=672
HEIGHT=384
FRAMES=124

while [ $# -gt 0 ]; do
  case "$1" in
    -p) PROMPT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for artifact in "$MODEL" "$NVFP4" "$UPSCALER" "$VAE"; do
  [ -e "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 3; }
done

cd "$ROOT" || exit 1
mkdir -p outputs scratch

# The release's Sol draft, as the clip tool sets it.
export H3_DIT_SOL=0.9

# One request per line: prompt, seed, steps, width, height, frames, output.
printf '%s\t%s\t%s\t%s\t%s\t%s\toutputs/resident-a.mp4\n' \
  "$PROMPT" "$SEED" "$STEPS" "$WIDTH" "$HEIGHT" "$FRAMES" > scratch/jobs.txt
printf '%s\t%s\t%s\t%s\t%s\t%s\toutputs/resident-b.mp4\n' \
  "$PROMPT" "$SEED" "$STEPS" "$WIDTH" "$HEIGHT" "$FRAMES" >> scratch/jobs.txt

guard() {
  local log=$1
  shift
  python3 tools/host_memory_guard.py --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 3600 \
    --log "scratch/$log.memory.jsonl" -- "$@"
}

# The seconds since `$1`, so both lifetimes carry a comparable number.
since() {
  awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN {printf "%.3f", b-a}'
}

echo "== resident: two requests, one process"
started=$(date +%s.%N)
guard resident "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
  --jobs scratch/jobs.txt --upscaler "$UPSCALER" 2>&1 | tee scratch/resident.out
[ "${PIPESTATUS[0]}" = "0" ] || exit 1
resident_seconds=$(since "$started")
echo "WALL stage=resident seconds=$resident_seconds"

echo "== two-process: stage 1, latent only"
started=$(date +%s.%N)
guard accept-stage1 "$ROOT/h3-rs" -d "$MODEL" --text-nvfp4 "$NVFP4" \
  -p "$PROMPT" --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
  --steps "$STEPS" --seed "$SEED" --save-latent scratch/accept.bin \
  --save-audio scratch/accept.pcm --latent-only 2>&1 | tee scratch/accept-stage1.out
[ "${PIPESTATUS[0]}" = "0" ] || exit 1
stage1_seconds=$(since "$started")
echo "WALL stage=two-process-stage1 seconds=$stage1_seconds"

echo "== two-process: stage 2, upscale and H3 decode"
started=$(date +%s.%N)
guard accept-stage2 "$ROOT/h3-sol-rs" --latent scratch/accept.bin \
  --audio scratch/accept.pcm --upscaler "$UPSCALER" --h3-vae "$VAE" \
  --h3-video upscaled --hash -o outputs/accept-legacy.mp4 \
  2>&1 | tee scratch/accept-stage2.out
[ "${PIPESTATUS[0]}" = "0" ] || exit 1
stage2_seconds=$(since "$started")
echo "WALL stage=two-process-stage2 seconds=$stage2_seconds"
echo "WALL two-process total=$(awk -v a="$stage1_seconds" -v b="$stage2_seconds" \
  'BEGIN {printf "%.3f", a+b}') resident-two-clips=$resident_seconds"

field() {
  # field <line> <name>: the `name=value` word's value.
  grep -o "$2=[^ ]*" <<<"$1" | head -1 | cut -d= -f2
}

line1=$(grep '^JOB n=1 ' scratch/resident.out)
line2=$(grep '^JOB n=2 ' scratch/resident.out)
legacy=$(grep -o 'H3-VAE frames_fnv=[0-9a-f]*' scratch/accept-stage2.out | cut -d= -f2)

first=$(field "$line1" frames_fnv)
second=$(field "$line2" frames_fnv)
status=0

echo "frames job1=$first job2=$second two-process=$legacy"

# Each marker is anchored to a word start: `load=` also appears inside
# `upscaler_load=`, and a hit on one must not stand in for the other.
for marker in ' text=[0-9.]*s \(hit\)' ' load=[0-9.]*s \(hit\)' \
  ' upscaler_load=[0-9.]*s \(hit\)'; do
  grep -Eq "$marker" <<<"$line2" || {
    echo "FAIL job 2 did not report $marker"
    status=1
  }
done

[ -n "$first" ] && [ "$first" = "$second" ] || {
  echo "FAIL job 1 and job 2 differ"
  status=1
}
[ -n "$first" ] && [ "$first" = "$legacy" ] || {
  echo "FAIL the resident route and the two-process route differ"
  status=1
}

if [ "$status" = "0" ]; then
  echo "RESULT PASS resident product route"
else
  echo "RESULT FAIL"
fi
exit "$status"
