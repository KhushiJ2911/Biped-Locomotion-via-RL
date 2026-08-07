#!/usr/bin/env bash
# Resolve the two remaining cross-embodiment ambiguities:
#   Berkeley at n=30 -- its n=10 ratio (1.66, CI [0.81,5.57]) is unresolved and
#     it is the only other robot whose trend pointed the same way as G1.
#   Op3 twice -- as shipped, and with dt/iterations matched to the other robots,
#     so its result stops being confounded by non-standard solver settings.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cool(){ echo "[cooldown] 120s (GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)C)"; sleep 120; }

echo "=== [1/3] Berkeley n=30  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env BerkeleyHumanoidJoystickFlatTerrain --duration 1.0 --seeds 30 --engines c jax warp
cool
echo "=== [2/3] Op3 n=30 as shipped (dt=0.004, iters=1)  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env Op3Joystick --duration 1.0 --seeds 30 --engines c jax warp --tag asshipped
cool
echo "=== [3/3] Op3 n=30 matched (dt=0.002, iters=3)  $(date -Is) ==="
$PY -u -m sim2sim.divergence --env Op3Joystick --duration 1.0 --seeds 30 --engines c jax warp \
    --base-timestep 0.002 --base-iterations 3 --tag matched
echo "=== RESOLVE QUEUE DONE $(date -Is) ==="
