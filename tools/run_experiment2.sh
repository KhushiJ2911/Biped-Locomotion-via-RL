#!/usr/bin/env bash
# Experiment 2: train N seeds in one backend, evaluate each in every backend.
#
# The design point: the cross-backend gap is only a finding if it exceeds
# ordinary seed-to-seed training noise. Training several seeds in a SINGLE
# backend and evaluating each in BOTH gives us both quantities from the same
# runs -- the null model and the effect -- without paying to train in both.
#
# Runs strictly serially with a cooldown between jobs. This is a 4GB laptop
# GPU, and back-to-back saturation for a day is what makes a laptop hot; the
# cooldown costs little relative to a multi-hour training run.
#
# Usage:
#   tools/run_experiment2.sh [TIMESTEPS] [SEEDS] [TRAIN_IMPL]
#   nohup setsid tools/run_experiment2.sh 50000000 3 jax > results/exp2.log 2>&1 &

set -uo pipefail

TIMESTEPS="${1:-50000000}"
NSEEDS="${2:-3}"
TRAIN_IMPL="${3:-jax}"
# Resume point. Training is the expensive part and checkpoints are named by
# seed, so restarting from 0 after an interruption would silently overwrite a
# finished policy with a fresh one. Evaluation still covers every seed.
START_SEED="${4:-0}"

ENVNAME="G1JoystickFlatTerrain"
PY="/home/keyu/miniconda3/envs/biped/bin/python"
ROOT="/home/keyu/Downloads/Biped-Locomotion-via-RL"
# 512 was measured faster than 1024 on this card (815 vs 655 steps/s): the GPU
# is already saturated at 512, so more envs only add memory traffic.
NUM_ENVS=512
NACONMAX=8192
COOLDOWN=180

cd "$ROOT" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== Experiment 2 ==="
echo "env=$ENVNAME impl=$TRAIN_IMPL timesteps=$TIMESTEPS seeds=$NSEEDS envs=$NUM_ENVS"
echo "started $(date -Is)"

gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "?"; }

# Wait until the GPU is below a threshold, so a hot card is never handed
# straight into the next multi-hour job.
cool_down() {
  echo "[cooldown] ${COOLDOWN}s (GPU $(gpu_temp)C)"
  sleep "$COOLDOWN"
  for _ in $(seq 1 60); do
    t=$(gpu_temp)
    [ "$t" = "?" ] && break
    [ "$t" -lt 60 ] 2>/dev/null && break
    echo "[cooldown] GPU still ${t}C, waiting"
    sleep 30
  done
  echo "[cooldown] done (GPU $(gpu_temp)C)"
}

for SEED in $(seq "$START_SEED" $((NSEEDS - 1))); do
  CKPT_EXISTING="checkpoints/${ENVNAME}_impl-${TRAIN_IMPL}_seed${SEED}"
  echo
  echo "--- TRAIN seed=$SEED impl=$TRAIN_IMPL  $(date -Is)  GPU $(gpu_temp)C ---"
  if [ -e "$CKPT_EXISTING" ]; then
    echo "    note: $CKPT_EXISTING exists and will be overwritten by this run"
  fi
  "$PY" -u -m sim2sim.train \
      --env "$ENVNAME" --impl "$TRAIN_IMPL" --seed "$SEED" \
      --timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" --naconmax "$NACONMAX"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! training seed=$SEED FAILED rc=$rc -- continuing to next seed"
    continue
  fi
  cool_down
done

echo
echo "=== EVALUATION ==="
for SEED in $(seq 0 $((NSEEDS - 1))); do
  CKPT="checkpoints/${ENVNAME}_impl-${TRAIN_IMPL}_seed${SEED}"
  if [ ! -e "$CKPT" ]; then
    echo "!! missing checkpoint $CKPT -- skipping"
    continue
  fi
  echo
  echo "--- EVAL seed=$SEED  $(date -Is) ---"
  # Long horizon on purpose: engine divergence compounds over time, and a
  # policy that terminates early never gives it the chance to show up.
  "$PY" -u -m sim2sim.evaluate \
      --env "$ENVNAME" --ckpt "$CKPT" \
      --episodes 128 --max-steps 1000 --naconmax "$NACONMAX" \
      --impls jax warp
  cool_down
done

echo
echo "=== DONE $(date -Is) ==="
echo "Cross-backend gaps:      results/eval_*.json"
echo "Seed-to-seed null model: compare return_mean across seeds within one backend"
