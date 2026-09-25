#!/bin/bash
#SBATCH --time=20:00:00

module load python/3.12.12 Anaconda3/conda-23.1.0

rm -r tau_initial_*

CONFIG=$(cd ../../src/fpinns && pwd)/config.yaml
PYTHONPATH_SRC=$(cd ../../src && pwd)

for i in {10..14..2}
do

mkdir -p tau_initial_$i
cd tau_initial_$i

echo "#!/bin/sh" > fixed_tau_$i.sh
echo "#SBATCH --time=24:00:00" >> fixed_tau_$i.sh
echo "module load python/3.10.7 Anaconda3/conda-23.1.0" >> fixed_tau_$i.sh
echo "export PYTHONPATH=$PYTHONPATH_SRC" >> fixed_tau_$i.sh
echo "python3 -u -m fpinns.main --config $CONFIG  --set training.tau_init=$i. --set fdm_signals.1.freq=8 --set fdm_signals.1.force_mag=1.0 --set fdm_signals.1.x0=0.5 --set fdm_signals.1.v0=0.0 > file_tau_$i.txt" >> fixed_tau_$i.sh


#sbatch --exclude=cn[001-00$((i/2+2))] fixed_tau_$i.sh
sbatch fixed_tau_$i.sh


cd ..

done

