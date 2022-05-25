#!/bin/bash
#SBATCH --mail-user=michel.hayoz@artorg.unibe.ch
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="deepLabv3"
#SBATCH --partition=gpu-invest
#SBATCH --account=ws_00000
##SBATCH --array=1-1


#### Your shell commands below this line ####

module load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate alleyoop

#param_store=porcine.txt     # args.txt contains 1000 lines with 2 arguments per line.

#sequence=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')    # Get first argument

cd scripts

python -u alleyoop_get_trajectory.py /storage/workspaces/artorg_aimi/ws_00000/innosuisse_surgical_robot/01_Datasets/05_slam/tum_rgbd/rgbd_dataset_freiburg1_xyz --device gpu --config ../configuration/alleyoop_slam_tum.yaml --outpath ../output/tum





