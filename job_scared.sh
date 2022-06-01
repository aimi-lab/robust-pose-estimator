#!/bin/bash
#SBATCH --mail-user=michel.hayoz@artorg.unibe.ch
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="deepLabv3"
#SBATCH --partition=gpu-invest
#SBATCH --account=ws_00000
#SBATCH --gres=gpu:1
#SBATCH --array=1-3


#### Your shell commands below this line ####

module load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate alleyoop

param_store=scared.txt     # args.txt contains 1000 lines with 2 arguments per line.

sequence=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')    # Get first argument

cd scripts

python -u alleyoop_get_trajectory.py ${sequence} --device gpu --outpath ${sequence}/data/alley_oop/weighted --log scared





