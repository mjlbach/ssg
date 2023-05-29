#!/bin/bash           
#
#SBATCH --job-name=directed_search_rgb_seed_0
#SBATCH --partition=viscam
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

source ~/.bashrc  
conda activate ssg

export GIBSON_ASSETS_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/assets
export IGIBSON_DATASET_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/ig_dataset

cd /sailhome/mjlbach/Repositories/ssg

python scripts/train.py ++experiment_name=$SLURM_JOB_NAME +experiment=neurips/directed_search/rgb
