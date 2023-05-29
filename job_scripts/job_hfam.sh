#!/bin/bash
#
#SBATCH --job-name=relational_search_rgb_10
#SBATCH --partition=viscam
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mjhwang@stanford.edu

source ~/.bashrc
conda activate ssg

export GIBSON_ASSETS_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/assets
export IGIBSON_DATASET_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/ig_dataset

cd /sailhome/mjhwang/Repositories/ssg

python scripts/train.py ++experiment_name=$SLURM_JOB_NAME +experiment=5_3_22/relational_search_sg_fixed_attention
