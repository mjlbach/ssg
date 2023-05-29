#!/bin/bash           
#                     
#SBATCH --job-name=search_test
#SBATCH --partition=viscam,svl
#SBATCH --exclude=viscam4
#SBATCH --time=48:00:00    
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=20    
#SBATCH --mem=30G    
#SBATCH --gres=gpu:3090:1    
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com
                      
source ~/.bashrc      
conda activate ssg    
                      
export GIBSON_ASSETS_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/assets
export IGIBSON_DATASET_PATH=/svl/u/mjlbach/Repositories/ig-data-bundle/ig_dataset

cd /sailhome/mjlbach/Repositories/ssg    
                      
python scripts/train.py ++experiment_name=$SLURM_JOB_NAME +experiment=4_17_22/choice_sg
