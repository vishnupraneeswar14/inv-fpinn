#!/bin/sh
#SBATCH --time=24:00:00
module load python/3.10.7 Anaconda3/conda-23.1.0
export PYTHONPATH=/home/co23btech11024.co.iith/fpinns/Vishnu/mini_project_sem7/inv-fpinn/src
python3 -u -m fpinns.main --config /home/co23btech11024.co.iith/fpinns/Vishnu/mini_project_sem7/inv-fpinn/src/fpinns/config.yaml --set training.tau_init=4. > file_tau_4.txt
