#!/bin/sh
#SBATCH --time=24:00:00
module load python/3.12.12 Anaconda3/conda-23.1.0
export PYTHONPATH=/home/co23btech11024.co.iith/fpinns/Vishnu/mini_project_sem7/inv-fpinn/src
python3 -u -m fpinns.main --config /home/co23btech11024.co.iith/fpinns/Vishnu/mini_project_sem7/inv-fpinn/src/fpinns/config.yaml --set training.mode=tau --set training.resume_path=/home/co23btech11024.co.iith/fpinns/Vishnu/mini_project_sem7/inv-fpinn/experiments/no_mask/lr_tau_4e-3/tau_initial_16/artifacts/checkpoints/alpha/step_6200.pth > resume_tau_16.txt
