base_model=$1
lora_weights=$2
test_file=$3
output_file=$4
lang_pair=$5

python hf_lora_generate.py \
    --base_model ${base_model} \
    --lora_weights ${lora_weights} \
    --test_file ${test_file} \
    --output_file ${output_file} \
    --batch_size 16 \
    --lang_pair ${lang_pair}