#!/bin/bash
#SBATCH --job-name=gnt_llm-critic
#SBATCH --output=./logs/%A.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu-h100
#SBATCH --partition=h100

source ~/mydata/venvs/building/bin/activate
echo `whereis python`

MODEL="gpt-4.1-2025-04-14"
MODEL_NAME="gpt-4.1-2025-04-14"

TRANSLATION_MODELS=( "Qwen2.5-72B-Instruct" "gemma-3-27b-it" "EuroLLM-9B-Instruct" )
# TRANSLATION_MODELS=( "Qwen2.5-72B-Instruct" )

CONFIG="CoT"
INPUT_FILE="data/data_release_MT_${CONFIG}.tsv"
OUTPUT_DIR="results/critic/${CONFIG}"
mkdir -p $OUTPUT_DIR

for tr in ${TRANSLATION_MODELS[@]}; do
    echo "Running GPT 4.1 critic for translation model: $tr and config: $CONFIG"

    OUTPUT_FILE="${OUTPUT_DIR}/critic_gender_label_${MODEL_NAME}-v3_${tr}.tsv"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file $OUTPUT_FILE already exists. Skipping..."
        continue
    fi

    python src/critic.py \
        --model_name_or_path $MODEL \
        --input_file $INPUT_FILE \
        --target_col $tr \
        --prompt_file "config/gender_detection-v3.prompt" \
        --output_file $OUTPUT_FILE
done
