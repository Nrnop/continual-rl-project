#!/bin/bash
# Run both agents over seeds 0-4 (sequential).
# For faster execution, parallelize across seeds/agents.

set -e

SEEDS="0 1 2 3 4"

echo "=== Running Vanilla PPO ==="
for s in $SEEDS; do
    echo "--- vanilla seed=$s ---"
    python -m src_continuous_control.train --agent vanilla --seed $s
done

echo "=== Running PT-PPO ==="
for s in $SEEDS; do
    echo "--- pt seed=$s ---"
    python -m src_continuous_control.train --agent pt --seed $s
done

echo "=== Plotting comparison ==="
python -m src_continuous_control.plots.plot_compare --seeds $SEEDS

echo "=== Done ==="
