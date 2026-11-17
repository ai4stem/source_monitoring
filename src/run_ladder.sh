
#!/usr/bin/env bash
# run_ladder.sh -- probe-only the size-ladder models for hidden-state analysis.
set -u
GROUP="${1:-qwen}"
NGPU="${NGPU:-4}"
CKPT_ROOT="outputs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS="--large_scale --probe_only --n_trivia 4000 --n_popqa 2000 --n_conflict 1500 --target_parametric 2500 --target_confab 800 --target_conflict 1000"

case "$GROUP" in
  qwen)   MODELS=(qwen3-1.7b qwen3-4b qwen3-14b) ;;
  others) MODELS=(llama32-1b llama32-3b mistral-7b mistral-nemo-12b) ;;
  full)   MODELS=(qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b) ;;
  all)    MODELS=(qwen3-1.7b qwen3-4b qwen3-14b llama32-1b llama32-3b mistral-7b mistral-nemo-12b qwen3-32b) ;;
  *) echo "unknown group: $GROUP"; exit 1 ;;
esac

is_distributed () {
  python3 -c "from registry import get_model
try:
    print(1 if get_model('$1').distributed else 0)
except Exception:
    print(0)"
}

SMALL=(); LARGE=()
for m in "${MODELS[@]}"; do
  if [ "$(is_distributed "$m")" = "1" ]; then LARGE+=("$m"); else SMALL+=("$m"); fi
done

echo "=== probe-only ladder: group=$GROUP  $(date) ==="
echo "  small: ${SMALL[*]:-none}"
echo "  large: ${LARGE[*]:-none}"

i=0
for m in "${SMALL[@]}"; do
  gpu=$(( i % NGPU ))
  echo ">>> probing $m on GPU $gpu (log: $CKPT_ROOT/probe_${m}.out) $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" python train_server.py --model "$m" $ARGS > "$CKPT_ROOT/probe_${m}.out" 2>&1 &
  i=$(( i + 1 ))
  [ $(( i % NGPU )) -eq 0 ] && wait
done
wait

for m in "${LARGE[@]}"; do
  echo ">>> probing $m (distributed) (log: $CKPT_ROOT/probe_${m}.out) $(date)"
  python train_server.py --model "$m" $ARGS > "$CKPT_ROOT/probe_${m}.out" 2>&1
done

echo "=== done $(date) ==="
for m in "${MODELS[@]}"; do
  f="$CKPT_ROOT/eval_${m}.jsonl"
  [ -f "$f" ] && echo "  $f ($(wc -l < "$f") items)" || echo "  $f MISSING"
done
