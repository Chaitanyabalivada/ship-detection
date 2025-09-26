#!/bin/bash
#SBATCH --job-name=yolo-ships
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Load modules (adjust to IITM HPC setup)
module load python/3.10 cuda/11.8 cudnn/8.6

cd $HOME/projects/ship-detection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python train_yolo.py --data ./data/ships.yaml --epochs 100 --batch 16 --imgsz 640 --output ./results
