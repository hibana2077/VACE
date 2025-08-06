#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=00:15:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info-v100.txt
source /scratch/rp06/sl5952/VACE/.venv/bin/activate

cd ..
# Run training
python train.py --config configs/cotton_r18.yaml >> out_train_v100.txt
