# Description: Script to run LoQT with 8-bit quantization on 350m model
# nproc_per_node 1 is number of GPUs per node and nnodes 1 is number of nodes
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main_Llama_different_dataset_kr.py \
    --model_config configs/llama_130m.json \
    --dataset_name HAERAE-HUB/KOREAN-WEBTEXT \
    --dataset_text_field text \
    --validation_split_percentage 5 \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
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
    --name 130m_LoQT_korean_webtext
    
