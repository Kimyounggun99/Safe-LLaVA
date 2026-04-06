#!/bin/bash
#



python -m llava.eval.model_vqa_loader \
    --model-path  ./checkpoints/Safe_LLaVA \
    --question-file ./playground/data/eval/PRISM/PRISM_refusal_soft.jsonl \
    --image-folder ./playground/data/eval/PRISM/biometric_images \
    --answers-file ./playground/data/eval/PRISM/answers/PRISM-soft-refusal-safe-llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.model_vqa_loader \
    --model-path  ./checkpoints/Safe_LLaVA \
    --question-file ./playground/data/eval/PRISM/PRISM_refusal_hard.jsonl \
    --image-folder ./playground/data/eval/PRISM/biometric_images \
    --answers-file ./playground/data/eval/PRISM/answers/PRISM-hard-refusal-safe-llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


cd ./playground/data/eval/PRISM



python GPT_soft_refusal_evaluation.py --mode Safe-LLaVA --API_key {Your-API-Key}
python GPT_hard_refusal_evaluation.py --mode Safe-LLaVA --API_key {Your-API-Key}