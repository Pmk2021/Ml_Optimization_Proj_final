#!/bin/bash

MODELS=(
    "finetuned_model"
    "lora_model"
    "bot_grad_2"
    "bot_grad_10"
    "bot_grad_20"
    "bot_grad_40"
    "top_grad_2"
    "top_grad_10"
    "top_grad_20"
    "top_grad_40"
    "bot_2_mag_size"
    "bot_10_mag_size"
    "bot_20_mag_size"
    "bot_40_mag_size"
    "big_2_mag_size"
    "big_10_mag_size"
    "big_20_mag_size"
    "big_40_mag_size"
)

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Running $model"
    echo "========================================"

    lm_eval \
        --model hf \
        --model_args pretrained="./${model}" \
        --tasks xwinograd_en,mmlu_social_sciences,babi,triviaqa,piqa \
        --device cuda:0 \
        --batch_size 4 \
        --limit 1000
done