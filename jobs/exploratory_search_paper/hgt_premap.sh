#!/bin/bash           
#
#SBATCH --job-name=hgt_premap_paper_seed_0
#SBATCH --partition=svl
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=90G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

source ~/.bashrc  
conda activate ssg

export GIBSON_ASSETS_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/assets
export IGIBSON_DATASET_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/ig_dataset

cd /sailhome/mjlbach/Repositories/ssg

python scripts/train.py ++experiment_name=$SLURM_JOB_NAME ++seed=0 +experiment=neurips/exploratory_search_paper/hgt_premap
