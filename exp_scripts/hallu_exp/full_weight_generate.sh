base_model=$1
test_file=$2
output_file=$3
lang_pair=$4

python hf_generate.py \
    --base_model ${base_model} \
    --test_file ${test_file} \
    --output_file ${output_file} \
    --lang-pair ${lang_pair} \
    --template_name llama2_chat \
    --batch_size 16