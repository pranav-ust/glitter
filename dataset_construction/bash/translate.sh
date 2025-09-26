#!/bin/bash
#SBATCH --job-name=BB_translate
#SBATCH --output=./logs/%A.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-debug
#SBATCH --partition=a6000

source ~/mydata/venvs/gnt/bin/activate

module load cuda

# # Tower 
# model="Unbabel/TowerInstruct-Mistral-7B-v0.2"
# output_file="./results/130225_sample_118_tower.tsv"
# prompt_template="tower"

# # DeepL
# model="deepl"

# # GPT-4o
# model="gpt-4o-2024-11-20"
# prompt_template="instruction"

# Tower API --> This is the model used in the final paper.
model="vesuvius"
output_file="./results/130225_remaining_${model}.tsv"

python src/translate.py \
    --dataset_file ./data/130225_remaining.tsv \
    --output_file $output_file \
    --model_name_or_path $model
    # --prompt_template $prompt_template