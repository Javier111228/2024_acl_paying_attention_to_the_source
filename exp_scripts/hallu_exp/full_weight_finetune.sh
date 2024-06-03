output_model=$1
train_data=$2
model_path=$3
lang_pair=$4
random_num=$(( RANDOM % 5001 ))
master_port=$(( random_num + 25000 ))
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi

cd /home/export/base/ycsc_chenkh/hitici_02/online1/LLM_for_mt/llama2-lora-fine-tuning

deepspeed --master_port ${master_port} finetune.py \
    --deepspeed ds_configs/stage2_no_offloading.conf \
    --model_name_or_path ${model_path} \
    --tokenizer_name ${model_path} \
    --train_files ${train_data} \
    --output_dir ${output_model} \
    --prompt_name llama2_chat\
    --lang_pair ${lang_pair} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --preprocessing_num_workers 1 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --use_fast_tokenizer true \
    --load_in_bits 16 \
    --evaluation_strategy  no \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 1e-2 \
    --dropout 0.3 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --bf16 True \
    --tf32 True \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 512 \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --ddp_timeout 3600 2>&1 | tee $(dirname "$output_model")/${lang_pair}.log
