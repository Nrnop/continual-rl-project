#!/bin/bash
# Quick smoke test: tiny run, frequent switch, no W&B.
# Verifies the training loop completes without errors for both agents.

set -e

echo "=== Smoke test: PT-PPO ==="
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb

echo "=== Smoke test: Vanilla PPO ==="
python -m src_continuous_control.train --agent vanilla --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb

echo "=== Smoke tests passed ==="
