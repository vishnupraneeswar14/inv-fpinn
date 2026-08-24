#!/bin/bash
#SBATCH --time=20:00:00

module load python/3.12.12 Anaconda3/conda-23.1.0

CONFIG=$(cd ../../../src/fpinns && pwd)/config.yaml
PYTHONPATH_SRC=$(cd ../../../src && pwd)

for i in {8..16..2}
do

cd tau_initial_$i

RESUME_PATH=$(pwd)/artifacts/checkpoints/alpha/step_6200.pth

echo "#!/bin/sh" > resume_tau_$i.sh
echo "#SBATCH --time=24:00:00" >> resume_tau_$i.sh
echo "module load python/3.12.12 Anaconda3/conda-23.1.0" >> resume_tau_$i.sh
echo "export PYTHONPATH=$PYTHONPATH_SRC" >> resume_tau_$i.sh
echo "python3 -u -m fpinns.main --config $CONFIG --set training.mode=tau --set training.resume_path=$RESUME_PATH > resume_tau_$i.txt" >> resume_tau_$i.sh

sbatch resume_tau_$i.sh

cd ..

done
