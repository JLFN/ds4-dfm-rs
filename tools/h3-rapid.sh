#!/bin/bash
# The Rapid-H3 path: a small draft, one learned upscale, one refine step.
#
# The two-stage route in one command. The DiT denoises a draft canvas at
# `--draft-scale` of the delivery canvas, a learned latent resizer lifts that
# draft's latent up to the delivery canvas, and one 0.72-sigma step refines it
# there. The soundtrack is the draft's own (`audio_frozen`) - the refine is
# video-only. This is the reference's route (`/data/Rapid-H3`), kept an arm and
# never the default: the route returns a *different shot* from the one pass.
#
# What it costs on this box at 1344x768 and five seconds (124 frames): 187-196 s
# warm on the wall against 347.6 s one-pass, and 235-252 s on a cold page cache.
# Two sessions' walls differ by 9 s while their stages agree to a percent, so
# read the wall as the clock state and the stages as the measurement
# (`docs/minimax-h3/draft-fp8-attention-2026-11-02.md` section 2).
#
# Usage: tools/h3-rapid.sh -p PROMPT [-o OUT] [options]
#          -p, --prompt TEXT      the prompt (or --prompt-file)
#          --prompt-file PATH     read the prompt from a file, newlines to spaces
#          -o, --output PATH      MP4 path (default outputs/rapid/rapid.mp4)
#          --seconds S            duration at 24 fps (default 5)
#          --width N              delivery width (default 1344)
#          --height N             delivery height (default 768)
#          --draft-scale F        draft canvas as a fraction of delivery (0.7)
#          --refine-sigma F       the refine's single-step sigma (0.72)
#          --steps N              draft denoise steps (default 4)
#          --seed N               seed (default 42)
#          --draft-attention K    the draft leg's data type: bf16 (default) or
#                                 e4m3; e4m3 measured -7.1% on the draft and
#                                 passes the clip gate, but moves the audio
#                                 1.55 dB, so it is asked for and never assumed
#          --crf N                x264 quality (default 15)
#          --preset NAME          x264 preset (default slow)
#          --one-pass             render without the route, for the comparison
#          --dry-run              print the command and the resolved geometry
#          -h, --help             this text
#
# Environment (set only when unset, so a caller's word wins):
#   H3_ROOT                 the checkout (default: the current directory)
#   H3_ARTIFACTS            checkpoint directory (default /mnt/h3-artifacts)
#   H3_MODEL                checkpoint root (default $H3_ARTIFACTS/MiniMax-H3-Turbo-v4)
#   H3_VAE                  the video decoder's source dir (the bf16i arm)
#   H3_UPSCALER_DIR         the learned latent resizer
#   H3_RS                   the renderer binary (default target/release/h3-rs)
#   H3_DIT_GEMM, H3_VAE_GEMM, H3_VAE_ATTENTION   the shipped arms
#
# Exit: 0 rendered, 1 the renderer failed, 2 bad arguments, 3 a missing artifact.
set -u

ROOT=${H3_ROOT:-$PWD}
A=${H3_ARTIFACTS:-/mnt/h3-artifacts}

PROMPT=
PROMPT_FILE=
OUTPUT=outputs/rapid/rapid.mp4
SECONDS_ARG=5
WIDTH=1344
HEIGHT=768
DRAFT_SCALE=0.7
REFINE_SIGMA=0.72
STEPS=4
SEED=42
DRAFT_ATTENTION=bf16
CRF=15
PRESET=slow
ONE_PASS=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--prompt) PROMPT=$2; shift 2 ;;
    --prompt-file) PROMPT_FILE=$2; shift 2 ;;
    -o|--output) OUTPUT=$2; shift 2 ;;
    --seconds) SECONDS_ARG=$2; shift 2 ;;
    --width) WIDTH=$2; shift 2 ;;
    --height) HEIGHT=$2; shift 2 ;;
    --draft-scale) DRAFT_SCALE=$2; shift 2 ;;
    --refine-sigma) REFINE_SIGMA=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --draft-attention) DRAFT_ATTENTION=$2; shift 2 ;;
    --crf) CRF=$2; shift 2 ;;
    --preset) PRESET=$2; shift 2 ;;
    --one-pass) ONE_PASS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -n "$PROMPT_FILE" ]; then
  [ -r "$PROMPT_FILE" ] || { echo "cannot read prompt file: $PROMPT_FILE" >&2; exit 2; }
  PROMPT=$(tr '\n' ' ' < "$PROMPT_FILE")
fi
[ -n "$PROMPT" ] || { echo "a prompt is required: -p TEXT or --prompt-file PATH" >&2; exit 2; }
case "$DRAFT_ATTENTION" in
  bf16|e4m3|fp8) ;;
  *) echo "--draft-attention must be bf16 or e4m3, not $DRAFT_ATTENTION" >&2; exit 2 ;;
esac

# Seconds become frames at 24 fps, then the release's 5 + 17k cadence
# (`h3_host.c:17-23`): five seconds asks for 120 and decodes 124, which is
# 5.167 s of video. The job line carries the request; the renderer aligns it.
FRAMES=$(awk -v s="$SECONDS_ARG" 'BEGIN {printf "%d", s * 24 + 0.5}')
if [ "$FRAMES" -lt 5 ]; then
  ALIGNED=5
else
  ALIGNED=$(( 5 + 17 * ( (FRAMES - 5 + 16) / 17 ) ))
fi

# The job line owns its geometry, so the request flags must agree with what is
# written here; the renderer refuses conflicting words.
MODEL=${H3_MODEL:-$A/MiniMax-H3-Turbo-v4}
: "${H3_VAE:=$A/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source}"
: "${H3_UPSCALER_DIR:=$A/models/latent_upscale}"
: "${H3_DIT_GEMM:=fp8}"
: "${H3_VAE_GEMM:=bf16i}"
: "${H3_VAE_ATTENTION:=tensor}"
export H3_VAE H3_UPSCALER_DIR H3_DIT_GEMM H3_VAE_GEMM H3_VAE_ATTENTION

BIN=${H3_RS:-$ROOT/target/release/h3-rs}
[ -x "$BIN" ] || BIN=$ROOT/h3-rs

for artifact in "$MODEL" "$H3_VAE" "$H3_UPSCALER_DIR"; do
  [ -e "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 3; }
done

case "$OUTPUT" in
  /*) OUT_PATH=$OUTPUT ;;
  *) OUT_PATH=$ROOT/$OUTPUT ;;
esac
mkdir -p "$(dirname "$OUT_PATH")"

# One request: prompt, seed, steps, width, height, frames, output.
JOBS=${OUT_PATH%.mp4}.jobs
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$PROMPT" "$SEED" "$STEPS" "$WIDTH" "$HEIGHT" "$FRAMES" "$OUT_PATH" > "$JOBS"

# The route's own flags. Without `--draft` the request is the one-pass arm.
ROUTE=()
if [ "$ONE_PASS" = "0" ]; then
  ROUTE=(--draft "$DRAFT_SCALE" --refine-sigma "$REFINE_SIGMA" \
    --draft-attention "$DRAFT_ATTENTION")
fi

draft_width=$(awk -v w="$WIDTH" -v s="$DRAFT_SCALE" 'BEGIN {printf "%d", (w*s/32)+0.5}')
draft_width=$(( draft_width * 32 ))
draft_height=$(awk -v h="$HEIGHT" -v s="$DRAFT_SCALE" 'BEGIN {printf "%d", (h*s/32)+0.5}')
draft_height=$(( draft_height * 32 ))

echo "PLAN route=$([ "$ONE_PASS" = "0" ] && echo two-stage || echo one-pass)"
echo "PLAN delivery=${WIDTH}x${HEIGHT} draft=${draft_width}x${draft_height} scale=$DRAFT_SCALE"
echo "PLAN frames=$FRAMES aligned=$ALIGNED seconds=$(awk -v f="$ALIGNED" 'BEGIN {printf "%.3f", f/24}') fps=24"
echo "PLAN steps=$STEPS seed=$SEED sigma=$REFINE_SIGMA draft-attention=$DRAFT_ATTENTION"
echo "PLAN arms H3_DIT_GEMM=$H3_DIT_GEMM H3_VAE_GEMM=$H3_VAE_GEMM H3_VAE_ATTENTION=$H3_VAE_ATTENTION"
echo "PLAN model=$MODEL"
echo "PLAN out=$OUT_PATH"

started=$(date +%s.%N)
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY $BIN -d $MODEL --jobs $JOBS ${ROUTE[*]} --crf $CRF --preset $PRESET"
  exit 0
fi

"$BIN" -d "$MODEL" --jobs "$JOBS" "${ROUTE[@]}" --crf "$CRF" --preset "$PRESET"
status=$?
if [ "$status" != "0" ]; then
  echo "FAIL the renderer exited $status" >&2
  exit 1
fi

wall=$(awk -v a="$started" -v b="$(date +%s.%N)" 'BEGIN {printf "%.3f", b-a}')
bytes=$(wc -c < "$OUT_PATH")
echo "CLIP out=$OUT_PATH bytes=$bytes wall=$wall"

if command -v ffprobe >/dev/null 2>&1; then
  duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,duration \
    -of default=nw=1:nk=1 "$OUT_PATH" 2>/dev/null | tr '\n' ' ')
  echo "CLIP ffprobe frames/duration=$duration"
fi
exit 0
