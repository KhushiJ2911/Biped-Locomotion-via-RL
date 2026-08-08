#!/usr/bin/env bash
# Experiment 5: policy evaluated in MuJoCo-C vs MJX, paired per episode.
# Three seeds so the cross-engine effect can be compared against seed noise,
# exactly as Experiment 2 did.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for S in 0 1 2; do
  echo "=== seed $S  $(date -Is) ==="
  $PY -u -m sim2sim.ceval \
      --ckpt "checkpoints/G1JoystickFlatTerrain_impl-jax_seed${S}" \
      --episodes 64 --max-steps 500 --impl jax
  echo "[cooldown]"; sleep 60
done
echo "=== CEVAL DONE $(date -Is) ==="
