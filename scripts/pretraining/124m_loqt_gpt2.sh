#!/bin/bash

# 현재 작업 디렉토리를 프로젝트 루트로 설정
# (파일명: torchrun_main.py가 있는 폴더에서 실행한다고 가정)

# variable 1. : update_proj_gap : default - 100
#     --use_hf_model \
#     --single_gpu \

torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main_gpt.py \
    --model_config configs/gpt2_124m.json \
    --seed 42 \
    --lr 0.001 \
    --rank 4 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --optimizer adamw \
    --use_loqt True \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --name gpt2_small_LoQT_lr_0001_rank_4
