#!/bin/bash
base_dir=path/to/directory/paying_attention_to_the_source

model_type=llama2
model_size=7b
base_model=path/to/model

data_dir=path/tp/data_dir
eval_data_dir=path/to/eval_data
train_data_dir=path/to/train_data
exp_output_dir=path/to/exp_output_dir
mkdir -p ${exp_output_dir}

language_pairs=(de-en en-de zh-en en-zh)

for language_pair in ${language_pairs[@]}
do
# lora finetune
model_output_path=${exp_output_dir}/lora_weight/${language_pair}
mkdir -p ${model_output_path}
bash ${base_dir}/exp_scripts/hallu_exp/lora_finetune.sh ${model_output_path} ${train_data_dir}/${language_pair}/train.${language_pair}.json ${base_model} ${language_pair}

# hallu dataset
test_file=${eval_data_dir}/hallu_dataset_${language_pair}.csv
output_dir=${exp_output_dir}/hallu_dataset
mkdir -p ${output_dir}
output_file=${output_dir}/${language_pair}.csv
bash ${base_dir}/exp_scripts/hallu_exp/vanilla_lora_generate.sh ${base_model} ${model_output_path} ${test_file} ${output_file} ${language_pair}

# flores testset
test_file=${eval_data_dir}/flores/${language_pair}/flores.${language_pair}.csv
output_dir=${exp_output_dir}/flores
mkdir -p ${output_dir}
output_file=${output_dir}/${language_pair}.csv
bash ${base_dir}/exp_scripts/hallu_exp/vanilla_lora_generate.sh ${base_model} ${model_output_path} ${test_file} ${output_file} ${language_pair}

# wmt22 testset
test_file=${eval_data_dir}/wmt22/${language_pair}/wmt22.${language_pair}.csv
output_dir=${exp_output_dir}/wmt22
mkdir -p ${output_dir}
output_file=${output_dir}/${language_pair}.csv
bash ${base_dir}/exp_scripts/hallu_exp/vanilla_lora_generate.sh ${base_model} ${model_output_path} ${test_file} ${output_file} ${language_pair}

done