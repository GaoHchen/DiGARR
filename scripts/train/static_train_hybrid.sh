ID=$1
CONFIG=$2
CUDA_VISIBLE_DEVICES=$ID python plenoxels/main.py --config-path plenoxels/configs/LLFF/llff_hybrid_$CONFIG.py