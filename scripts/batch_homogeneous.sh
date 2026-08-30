#!/usr/bin/env bash
# Run N homogeneous games for one or more models.
#
# Usage:
#   ./scripts/batch_homogeneous.sh                                    # all models, 30 seeds
#   ./scripts/batch_homogeneous.sh -m gpt5.5                          # single model
#   ./scripts/batch_homogeneous.sh -m gpt5.5 -n 10                    # 10 games
#   ./scripts/batch_homogeneous.sh -m gpt5.5 -s 11 -n 10              # seeds 11-20
#   ./scripts/batch_homogeneous.sh -m gpt5.5,claude_opus4.7 -n 5
#   ./scripts/batch_homogeneous.sh -t -m gpt5.5                       # text-only ablation
set -euo pipefail

ALL_MODELS="gpt5.5 claude_opus4.7 gemini3.1pro"
NUM_GAMES=30
START_SEED=1
MODELS=""
EXTRA_ARGS=""
TEXT_ONLY=0
# Text-only ablation runs are written under this base so they stay cleanly
# separated from the default image+text runs in game_logs/. (Same timestamped
# subdirs, just a different root, so evaluate_batch.py can be pointed at each
# condition independently.)
TEXT_ONLY_OUTPUT_BASE="game_logs/text_only"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- EXTRA_HYDRA_ARGS...]

Options:
  -m MODELS   Comma-separated model names (default: all models)
  -n NUM      Number of games per model (default: $NUM_GAMES)
  -s START    Starting seed (default: $START_SEED)
  -t          Text-only ablation: disable images (send_images=false) and
              write runs under $TEXT_ONLY_OUTPUT_BASE/ to keep them separate
              from the default image+text runs
  -h          Show this help

Examples:
  $(basename "$0")                                     # all 3 models × 30 seeds
  $(basename "$0") -m gpt5.5 -n 10                     # GPT-5.5 × 10 seeds
  $(basename "$0") -m gpt5.5,claude_opus4.7 -n 5       # two models × 5 seeds
  $(basename "$0") -m gpt5.5 -n 10 -- video=false      # pass extra Hydra args
  $(basename "$0") -t -m gpt5.5                        # text-only ablation, 30 seeds
EOF
    exit 0
}

while getopts "m:n:s:th" opt; do
    case $opt in
        m) MODELS="${OPTARG//,/ }" ;;
        n) NUM_GAMES="$OPTARG" ;;
        s) START_SEED="$OPTARG" ;;
        t) TEXT_ONLY=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
EXTRA_ARGS="$*"

if [ -z "$MODELS" ]; then
    MODELS="$ALL_MODELS"
fi

# Text-only ablation: prepend the ablation overrides so any user-supplied
# EXTRA_ARGS still win (Hydra applies later overrides last).
MODE="image+text"
if [ "$TEXT_ONLY" -eq 1 ]; then
    MODE="TEXT-ONLY (send_images=false)"
    EXTRA_ARGS="send_images=false output_base=$TEXT_ONLY_OUTPUT_BASE $EXTRA_ARGS"
fi

END_SEED=$((START_SEED + NUM_GAMES - 1))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  QUACK Homogeneous Batch Runner"
echo "============================================"
echo "Models:     $MODELS"
echo "Seeds:      $START_SEED .. $END_SEED ($NUM_GAMES games each)"
echo "Input mode: $MODE"
[ -n "$EXTRA_ARGS" ] && echo "Extra args: $EXTRA_ARGS"
echo "--------------------------------------------"

total=0
failed=0

for model in $MODELS; do
    echo ""
    echo ">>> Model: $model"
    for seed in $(seq "$START_SEED" "$END_SEED"); do
        total=$((total + 1))
        echo -n "  [${model}] seed=${seed} ... "
        if python "$SCRIPT_DIR/run_game.py" model="$model" seed="$seed" $EXTRA_ARGS \
            > /dev/null 2>&1; then
            echo "OK"
        else
            echo "FAILED"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "============================================"
echo "  Done: $((total - failed))/$total succeeded"
[ "$failed" -gt 0 ] && echo "  $failed game(s) failed"
echo "============================================"
