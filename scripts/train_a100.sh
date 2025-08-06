#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=00:45:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info-a100.txt
source /scratch/rp06/sl5952/VACE/.venv/bin/activate

cd ..
# Run training
# python train.py --config configs/cotton_convnext_tiny.yaml >> out_train_a100.txt
# python train.py --config configs/cotton_r50.yaml >> out_train_a100.txt
python train.py --config configs/cotton_tiny_vit.yaml >> out_train_a100.txt