#!/usr/bin/env bash
# run_all.sh -- end-to-end: freeze data -> train all open models -> compare all.
#
# Usage (from the src/ directory):
#   nohup bash run_all.sh > run_all.out 2>&1 &
#   tail -f run_all.out        # watch progress
#
# Set TRAIN=0 to SKIP training (evaluate/compare only -- useful before the
# train_server.py loop is filled in, or to re-score with existing checkpoints).
#
# Requires (on the server):
#   pip install torch transformers peft accelerate bitsandbytes datasets \
#               openai anthropic numpy pandas scikit-learn
#   cp ../.env.example ../.env   &&  edit ../.env with your keys
#   (keys are read automatically from .env -- no need to export them)

set -u  # error on unset vars; do NOT set -e so one model failing won't kill all

TRAIN="${TRAIN:-1}"                 # 1 = train open models, 0 = skip training
LARGE_SCALE="${LARGE_SCALE:-1}"     # 1 = big external data (TriviaQA/PopQA/ConflictQA)
CKPT_ROOT="outputs"                 # where per-model checkpoints land
NGPU="${NGPU:-4}"                   # GPUs to spread small models over

# probing pool sizes (oversample; probing filters down to the targets below)
N_TRIVIA="${N_TRIVIA:-4000}"       # parametric-leaning source
N_POPQA="${N_POPQA:-2000}"         # confabulated-leaning source (fact total ~6000)
N_CONFLICT="${N_CONFLICT:-1500}"
T_PARAM="${T_PARAM:-2500}"
T_CONFAB="${T_CONFAB:-800}"
T_CONFLICT="${T_CONFLICT:-1000}"

# Reduce CUDA memory fragmentation (helps the last-step allocations).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Small models each fit on ONE GPU -> run several in parallel (one per GPU).
# Large models (distributed in registry, e.g. qwen3-32b) shard across all GPUs,
# so they run ALONE afterward.
SMALL_MODELS=(qwen3-8b gemma3-12b phi4-mini llama3-8b-legacy phi3-mini-legacy)
LARGE_MODELS=(qwen3-32b)

echo "=================================================="
echo " source-monitoring full run   $(date)"
echo " TRAIN=$TRAIN  NGPU=$NGPU  fact pool: trivia=$N_TRIVIA popqa=$N_POPQA"
echo " targets: param=$T_PARAM confab=$T_CONFAB conflict=$T_CONFLICT"
echo "=================================================="

python -c "from distributed import gpu_report; print(gpu_report())" 2>/dev/null || true

LS_FLAG=""
[ "$LARGE_SCALE" = "1" ] && LS_FLAG="--large_scale"
SCALE_ARGS="--n_trivia $N_TRIVIA --n_popqa $N_POPQA --n_conflict $N_CONFLICT \
--target_parametric $T_PARAM --target_confab $T_CONFAB --target_conflict $T_CONFLICT"

train_one () {   # $1 = model key, $2 = gpu id or "auto"
  local m="$1" gpu="$2"
  if [ "$gpu" = "auto" ]; then
    echo ">>> training $m (distributed, all GPUs)   $(date)"
    python train_server.py --model "$m" --ckpt_dir "$CKPT_ROOT/ckpt_$m" \
      $LS_FLAG $SCALE_ARGS
  else
    echo ">>> training $m on GPU $gpu   $(date)  (log: $CKPT_ROOT/train_${m}.out)"
    CUDA_VISIBLE_DEVICES="$gpu" python train_server.py --model "$m" \
      --ckpt_dir "$CKPT_ROOT/ckpt_$m" $LS_FLAG $SCALE_ARGS \
      > "$CKPT_ROOT/train_${m}.out" 2>&1
  fi
}

if [ "$TRAIN" = "1" ]; then
  # small models: launch in waves of NGPU, one model per GPU (parallel)
  i=0
  for m in "${SMALL_MODELS[@]}"; do
    gpu=$(( i % NGPU ))
    train_one "$m" "$gpu" &
    i=$(( i + 1 ))
    [ $(( i % NGPU )) -eq 0 ] && wait   # wait when all GPUs are busy
  done
  wait   # finish the last partial wave

  # large models: run alone, sharded across all GPUs
  for m in "${LARGE_MODELS[@]}"; do
    train_one "$m" "auto"
  done
else
  echo ">>> TRAIN=0 -> skipping training"
fi

# --- 3. evaluate + compare ALL registered models on the frozen data ---------
# For weights models we point at each model's own checkpoint if it exists;
# run_comparison loads per-model ckpt by key automatically when --ckpt_root set.
echo ">>> [3/3] running comparison across ALL models   $(date)"
python run_comparison.py --models all --ckpt_root "$CKPT_ROOT"

echo ">>> aggregating final table"
python run_comparison.py --aggregate

echo "=================================================="
echo " DONE  $(date)"
echo " results: outputs/comparison.json  (+ outputs/<model>/results.json)"
echo "=================================================="
