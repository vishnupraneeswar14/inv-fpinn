#!/bin/bash
#SBATCH --time=20:00:00

module load python/3.12.12 Anaconda3/conda-23.1.0

rm -r tau_actual_*

CONFIG=$(cd ../../../src/fpinns && pwd)/config.yaml
PYTHONPATH_SRC=$(cd ../../../src && pwd)

for ta in {6..10..2}
do

mkdir -p tau_actual_$ta
cd tau_actual_$ta

for ti in {4..14..5}
do

mkdir -p tau_initial_$ti
cd tau_initial_$ti

echo "#!/bin/sh" > fixed_tau_${ta}_${ti}.sh
echo "#SBATCH --time=24:00:00" >> fixed_tau_${ta}_${ti}.sh
echo "module load python/3.10.7 Anaconda3/conda-23.1.0" >> fixed_tau_${ta}_${ti}.sh
echo "export PYTHONPATH=$PYTHONPATH_SRC" >> fixed_tau_${ta}_${ti}.sh
echo "python3 -u -m fpinns.main --config $CONFIG --set physics.fdm_exp=true --set physics.tau_actual=$ta. --set training.tau_init=$ti. --set training.k_c_trainable=false > file_tau_${ta}_${ti}.txt" >> fixed_tau_${ta}_${ti}.sh


sbatch fixed_tau_${ta}_${ti}.sh


cd ..

done

cd ..

done