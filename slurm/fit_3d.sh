#!/bin/bash
#SBATCH --job-name=fit_3d
#SBATCH --output=/home/mbassler/slurm_logs/cancer_ml/%x_%j.log
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

echo "---START---"
date; pwd; hostname;
echo "$TMPDIR"

#load modules
echo "---LOADING MODULES---"
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

#install packages
echo "---INSTALLING PACKAGES---"
pip install --user pandas matplotlib numpy scipy keras_hub
pip install --user -e "$HOME"/github/cancer_ml

#COpy files
echo "---COPYING FILES---"
mkdir -p "$TMPDIR"/data
cp -r "$HOME"/data/cancer/train_100 "$TMPDIR"/data

#Run very simple script
echo "---RUNNING PYTHON SCRIPT---"
python "$HOME"/github/cancer_ml/scripts/cluster/naive_segment.py \
  --data_dir "$TMPDIR"/data/train_100 \
  --output_dir "$HOME"/output/cancer_ml \
  --tb_dir "$HOME"/output/cancer_ml/tb_runs

#log end
echo "---COMPLETED---"