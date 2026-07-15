#!/usr/bin/env bash
set -e

gpuid=${1:-0}
config_path=${2:-config/semanticposs_learnable_offset_b1_full.yaml}
mkdir -p logs_dir model_load_dir model_save_dir

CUDA_VISIBLE_DEVICES=${gpuid} python -u train_cylinder_asym.py \
  --config_path "${config_path}" \
  2>&1 | tee "logs_dir/semanticposs_$(date +%Y%m%d_%H%M%S).log"
