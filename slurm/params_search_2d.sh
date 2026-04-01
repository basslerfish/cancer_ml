#!/bin/bash
#SBATCH --job-name=params_search_2d
#SBATCH --output=/home/mbassler/slurm_logs/cancer_ml/%x_%j.log
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00

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
pip install --user pandas matplotlib numpy scipy keras-hub keras-tuner scikit-learn
pip install --user -e "$HOME"/github/cancer_ml

#Copy files
echo "---COPYING FILES---"
mkdir -p "$TMPDIR"/data
cp -r "$HOME"/data/cancer/2d/samples500_zscore_val15_test15_128-128 "$TMPDIR"/data
ls "$TMPDIR"/data

#Run very simple script
echo "---RUNNING PYTHON SCRIPT---"
python "$HOME"/github/cancer_ml/scripts/2d/cluster/search_hyperparams.py \
  --data_dir "$TMPDIR"/data/samples500_zscore_val15_test15_128-128 \
  --output_dir "$HOME"/output/cancer_ml/2d/ \

#log end
echo "---COMPLETED---"