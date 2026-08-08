#!/usr/bin/env bash
# Experiment 7: does the engine gap vanish as the solver converges?
#
# Op3 showed that matching dt/iterations turned a 2/14 null into 14/14
# significant, so solver precision clearly controls how visible the discrepancy
# is. This asks the sharper question: refine the solver as far as practical and
# see whether the gap tends to zero (a discretisation artefact that careful
# settings remove) or plateaus (a fixed implementation difference).
#
# Baseline condition only (--quick); the grid is over integration settings.
# Reuses the validated divergence sweep rather than new code.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for DT in 0.004 0.002 0.001 0.0005; do
  for IT in 1 3 10 50; do
    TAG="conv_dt${DT}_it${IT}"
    if [ -f "results/divergence_G1JoystickFlatTerrain_${TAG}.json" ]; then
      echo "=== skip $TAG (already done) ==="; continue
    fi
    echo "=== dt=$DT iterations=$IT  $(date -Is) ==="
    $PY -u -m sim2sim.divergence --env G1JoystickFlatTerrain --quick \
        --duration 1.0 --seeds 20 --engines c jax warp --njmax 512 \
        --base-timestep "$DT" --base-iterations "$IT" --tag "$TAG" --no-wandb
  done
done
echo "=== CONVERGENCE DONE $(date -Is) ==="
