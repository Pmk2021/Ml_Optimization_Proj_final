#!/bin/bash

set -e

echo "=== Full Fine-Tuning ==="
python scripts/finetune.py

echo "=== LoRA Fine-Tuning ==="
python scripts/finetune_lora.py

DENSITIES=(0.02 0.1 0.2 0.4)

for DENSITY in "${DENSITIES[@]}"; do
    echo "=== Smallest Weights (density=${DENSITY}) ==="
    python scripts/finetune_smallest.py --density "$DENSITY"

    echo "=== Smallest Gradient Weights (density=${DENSITY}) ==="
    python scripts/finetune_smallest_grad.py --density "$DENSITY"

    echo "=== Biggest Weights (density=${DENSITY}) ==="
    python scripts/finetune_biggest.py --density "$DENSITY"

    echo "=== Biggest Gradient Weights (density=${DENSITY}) ==="
    python scripts/finetune_biggest_grad.py --density "$DENSITY"
done

echo "=== All experiments completed ==="