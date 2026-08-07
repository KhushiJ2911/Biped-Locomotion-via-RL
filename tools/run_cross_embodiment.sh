#!/usr/bin/env bash
# Experiment 1 replicated across every biped Playground ships.
# Sequential by design: one GPU, and concurrent runs would contend for it and
# for the thermal budget. Identical protocol to the G1 run so the numbers are
# directly comparable -- same duration, same seeds, same engines.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for ENV in T1JoystickFlatTerrain BerkeleyHumanoidJoystickFlatTerrain Op3Joystick; do
  echo; echo "=== $ENV  $(date -Is)  GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)C ==="
  $PY -u -m sim2sim.divergence --env "$ENV" --duration 1.0 --seeds 10 --engines c jax warp
  rc=$?
  [ $rc -ne 0 ] && echo "!! $ENV FAILED rc=$rc -- continuing"
  echo "[cooldown] 120s"; sleep 120
done
echo; echo "=== CROSS-EMBODIMENT DONE $(date -Is) ==="
