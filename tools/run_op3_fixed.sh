#!/usr/bin/env bash
# Re-run Op3 with an explicit constraint budget. Every previous Op3 sweep is
# invalid: the default njmax was too small, so the solver dropped constraints
# ("nefc overflow", needs >=76) and returned corrupted trajectories. njmax=512
# is well clear of the requirement.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== [1/2] Op3 n=30 as shipped, njmax=512  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env Op3Joystick --duration 1.0 --seeds 30 \
    --engines c jax warp --njmax 512 --tag fixed_asshipped
echo "[cooldown]"; sleep 120
echo "=== [2/2] Op3 n=30 matched dt/iters, njmax=512  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env Op3Joystick --duration 1.0 --seeds 30 \
    --engines c jax warp --njmax 512 --base-timestep 0.002 --base-iterations 3 \
    --tag fixed_matched
echo "=== OP3 REDO DONE $(date -Is) ==="
