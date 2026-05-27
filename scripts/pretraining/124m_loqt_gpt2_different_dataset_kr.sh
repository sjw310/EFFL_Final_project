#!/bin/bash

# 현재 작업 디렉토리를 프로젝트 루트로 설정
# (파일명: torchrun_main.py가 있는 폴더에서 실행한다고 가정)

torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main_gpt_different_dataset.py \
    --model_config configs/gpt2_355m.json \
    --dataset_name HAERAE-HUB/KOREAN-WEBTEXT \
    --dataset_text_field text \
    --validation_split_percentage 5 \
    --use_hf_model \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --use_loqt True \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --single_gpu \
    --optimizer adamw \
    --name gpt2_medium_LoQT_4bit_korean_webtext
