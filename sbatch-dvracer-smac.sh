#!/bin/bash -l

# Read Settings file
# source settings.sh

echo "Environment:"         $ENV
echo "Model:"               $MODEL
echo "RUN:"					$RUN
echo "NN:"                  FNN

RUNPATH=${SCRATCH}/smac/${ENV}_${MODEL}/${RUN}/
mkdir -p $RUNPATH

cp run-dvracer.py $RUNPATH
cp readRewardHist.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=smac_${ENV}
#SBATCH --output=smac_${ENV}_${MODEL}_${RUN}_%j.out
#SBATCH --error=smac_${ENV}_${MODEL}_${RUN}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python3 run-dvracer.py --env "$ENV" --model "$MODEL" --run $RUN
python3 koralir.rlview --dir ./_results/ --out dvracer.png
python3 readRewardHist.py -fn ./_results/testingRewardHistory.npz -pte 20
EOF

chmod 755 run.sbatch
sbatch run.sbatch
