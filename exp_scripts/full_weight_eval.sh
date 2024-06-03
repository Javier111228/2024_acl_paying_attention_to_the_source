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

language_pairs=(en-de de-en en-zh zh-en)
for language_pair in ${language_pairs[@]}
do
# full weight finetune
model_output_path=${exp_output_dir}/full_weight/${language_pair}
mkdir -p ${model_output_path}
bash ${base_dir}/exp_scripts/hallu_exp/full_weight_finetune.sh ${model_output_path} ${train_data_dir}/${language_pair}/train.${language_pair}.json ${base_model} ${language_pair}

# infer
test_file=${eval_data_dir}/hallu_dataset_${language_pair}.csv
output_dir=${exp_output_dir}/hallu_dataset
mkdir -p ${output_dir}
output_file=${output_dir}/${language_pair}.csv
bash ${base_dir}/exp_scripts/hallu_exp/full_weight_generate.sh ${model_output_path} ${test_file} ${output_file} ${language_pair}

done