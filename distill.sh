#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=distill-flant5-sympathy
#SBATCH --gres=gpu:rtx_a6000:1 
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=f.samir@utoronto.ca
#SBATCH --output=R-flant5-%x.%j.out
#SBATCH --error=R-flant5-%x.%j.err
source ~/sympathy_venv/bin/activate
cd sympathy-in-media

python main_distillation.py distill-flant5