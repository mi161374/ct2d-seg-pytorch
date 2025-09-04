#!/bin/bash
#SBATCH --account=def-gduque
#SBATCH --job-name=ct2d_unet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/project/def-gduque/mi161374/ct2d-seg-pytorch/slurm/%x-%j.out

set -euo pipefail

# --- Modules & venv ---
module load python/3.12.4 cuda cudnn opencv
source $PROJECT/venvs/ct2d_seg/bin/activate
export PYTHONUNBUFFERED=1

# --- Data paths (SCRATCH) ---
export DATA_ROOT=/home/mi161374/scratch/ctseg_v1
export IMAGES="$DATA_ROOT/images"
export MASKS="$DATA_ROOT/masks_indexed"
export SPLITS="$DATA_ROOT/splits.json"

# --- Output dir (PROJECT) ---
RUN_NAME=unet_bs4_lr5e4_$(date +%Y%m%d_%H%M%S)
export OUTDIR=$PROJECT/ct2d-seg-pytorch/outputs/${RUN_NAME}
mkdir -p "$OUTDIR" "$OUTDIR/ckpts"

# --- W&B (offline on compute; sync later from a machine with internet) ---
export WANDB_MODE=offline
export WANDB_DIR="$OUTDIR/wandb"
# If you have a key and plan to run online from a login node later:
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx

echo "JOB $SLURM_JOB_ID starting on $(hostname)"
nvidia-smi || true
python - <<'PY'
import torch, torchvision, cv2, monai
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "CUDA build", torch.version.cuda)
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("monai", monai.__version__)
PY

cd $PROJECT/ct2d-seg-pytorch

# --- Train ---
python src/train.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --splits "$SPLITS" \
  --num_classes 8 \
  --epochs 80 \
  --bs 4 \
  --img_size 512 \
  --lr 5e-4 \
  --amp \
  --out "$OUTDIR" \
  --wandb

echo "DONE. Outputs in $OUTDIR"
