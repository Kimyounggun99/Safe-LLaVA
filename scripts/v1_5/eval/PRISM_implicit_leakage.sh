#!/bin/bash


python -m llava.eval.model_vqa_loader \
    --model-path  ./checkpoints/Safe_LLaVA \
    --question-file ./playground/data/eval/PRISM/PRISM_implicit_leakage.jsonl \
    --image-folder ./playground/data/eval/PRISM/biometric_images \
    --answers-file ./playground/data/eval/PRISM/answers/PRISM-implicit_leakage-safe-llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/PRISM

python GPT_implicit_leakage_evaluation.py --mode Safe-LLaVA --API_key {Your-API-Key}

