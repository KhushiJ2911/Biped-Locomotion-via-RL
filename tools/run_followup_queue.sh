#!/usr/bin/env bash
# Runs after the cross-embodiment sweeps, in priority order:
#   1. T1 at n=30 -- its n=10 error bars (75% relative) cannot resolve whether
#      the G1 Warp advantage replicates. This is the least secure claim in the
#      paper, so it gets compute first.
#   2. Noise floor at Experiment 2's exact settings.
#   3. Dose-response across all three policy seeds.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Wait for the cross-embodiment run to finish before touching the GPU.
while pgrep -f "run_cross_embodimen[t]\.sh" >/dev/null; do sleep 60; done
echo "=== cross-embodiment finished; follow-up queue starting $(date -Is) ==="

cool() { echo "[cooldown] 120s (GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)C)"; sleep 120; }

echo; echo "=== [1/3] T1 at n=30 seeds  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env T1JoystickFlatTerrain --duration 1.0 --seeds 30 --engines c jax warp
cool

echo; echo "=== [2/3] noise floor at Exp2 settings  $(date -Is) ==="
$PY -u tools/measure_noise_floor.py \
    --ckpt checkpoints/G1JoystickFlatTerrain_impl-jax_seed0 \
    --episodes 16 64 128 --max-steps 1000 --repeats 3
cool

echo; echo "=== [3/3] dose-response, 3 seeds  $(date -Is) ==="
for S in 0 1 2; do
  echo "--- dose-response seed=$S ---"
  $PY -u -m sim2sim.doseresponse \
      --env G1JoystickFlatTerrain \
      --ckpt "checkpoints/G1JoystickFlatTerrain_impl-jax_seed${S}" \
      --episodes 64 --max-steps 500
  cool
done

echo; echo "=== FOLLOW-UP QUEUE DONE $(date -Is) ==="
