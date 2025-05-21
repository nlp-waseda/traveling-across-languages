#!/bin/bash

# Define the list of subsets
subsets=(
  # xm3600_ar
  # xm3600_bn
  # xm3600_cs
  # xm3600_da
  # xm3600_de
  # xm3600_el
  # xm3600_en
  # xm3600_es
  # xm3600_fa
  # xm3600_fi
  # xm3600_fil
  # xm3600_fr
  # xm3600_hi
  # xm3600_hr
  # xm3600_hu
  # xm3600_id
  # xm3600_it
  # xm3600_he
  # xm3600_ja
  # xm3600_ko
  # xm3600_mi
  # xm3600_nl
  xm3600_no
  # xm3600_pl
  # xm3600_pt
  # xm3600_quz
  # xm3600_ro
  # xm3600_ru
  # xm3600_sv
  # xm3600_sw
  # xm3600_te
  # xm3600_th
  # xm3600_tr
  # xm3600_vi
  # xm3600_zh
)

# Create the log directory if it doesn't exist
mkdir -p xm3600_eval_log

# Loop over each subset and submit a job
for subset in "${subsets[@]}"; do
  sbatch <<EOT
#!/bin/bash
set -e -x

#SBATCH --job-name=llava_eval_${subset}
#SBATCH --output=./xm3600_eval_log/ours-v0.3-${subset}.log
#SBATCH --error=./xm3600_eval_log/ours-v0.3-${subset}.log
#SBATCH --gres=gpu:A6000:1  # Request one A6000 GPU
#SBATCH --cpus-per-task=8    # Request 8 CPU cores
#SBATCH --mem=32G            # Request 32GB of memory
#SBATCH --time=1-00:00:00    # Set a time limit of 1 day

source activate llava
cd lmms-eval-mmllm
echo ${subset}

python -m accelerate.commands.launch \
    --num_processes=1 \
    lmms_eval \
    --model=llava \
    --model_args=pretrained=/data/tir/projects/tir4/corpora/mmllm/checkpoints/llava-1.5-vicuna-7b-v0.3/checkpoint-30000/ \
    --tasks=${subset} \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix=ours-v0.3 \
    --output_path=./logs/

EOT
done
