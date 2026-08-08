#!/usr/bin/env bash
# Wait for the ceval run to clear, then: precision test, then convergence grid.
set -uo pipefail
PY=/home/keyu/miniconda3/envs/biped/bin/python
cd /home/keyu/Downloads/Biped-Locomotion-via-RL || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f "sim2sim\.ceva[l]" >/dev/null; do sleep 30; done
echo "=== [1/3] precision float32  $(date -Is) ==="
$PY -u -m sim2sim.precision --seeds 20
echo "=== [2/3] precision float64  $(date -Is) ==="
$PY -u -m sim2sim.precision --seeds 20 --x64
echo "=== [3/3] convergence grid  $(date -Is) ==="
tools/run_convergence.sh
echo "=== EXP 6+7 DONE $(date -Is) ==="
