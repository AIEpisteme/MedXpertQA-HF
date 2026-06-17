#!/bin/bash
set -e

model=${1:-"EpistemeAI/Reasoning-Medical-27B"}
dataset=${2:-"medxpertqa_sampled"}
task=${3:-"text"}
output_dir=${4:-"hf-dev"}

method=${5:-"zero_shot"}
prompting_type=${6:-"cot"}
max_samples=${7:--1}
max_new_tokens=${8:-4096}

device_map=${HF_DEVICE_MAP:-"auto"}
torch_dtype=${HF_TORCH_DTYPE:-"auto"}
image_policy=${HF_IMAGE_POLICY:-"error"}

date +"%Y-%m-%d %H:%M:%S"
echo "Provider: hf"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Task: ${task}"
echo "Output: ${output_dir}"

log_dir="outputs/${output_dir}/${model}/${dataset}/${method}/${prompting_type}/logs"
mkdir -p "${log_dir}"

cp "${BASH_SOURCE[0]}" "${log_dir}/run_hf.sh"
cp main.py "${log_dir}/main.py"
cp utils.py "${log_dir}/utils.py"
cp setup.py "${log_dir}/setup.py"
cp model/hf_agent.py "${log_dir}/hf_agent.py"
cp config/prompt_templates.py "${log_dir}/prompt_templates.py"

python main.py \
    --provider hf \
    --model "${model}" \
    --dataset "${dataset}" \
    --task "${task}" \
    --output-dir "${output_dir}" \
    --method "${method}" \
    --prompting-type "${prompting_type}" \
    --max-samples "${max_samples}" \
    --num-threads 1 \
    --hf-device-map "${device_map}" \
    --hf-torch-dtype "${torch_dtype}" \
    --hf-max-new-tokens "${max_new_tokens}" \
    --hf-image-policy "${image_policy}"

# bash scripts/run_hf.sh
