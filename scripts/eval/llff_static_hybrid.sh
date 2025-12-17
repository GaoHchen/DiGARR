ID=$1
logdir=$2
CONFIG=$3
CUDA_VISIBLE_DEVICES=$ID python plenoxels/main.py --config-path plenoxels/configs/LLFF/llff_hybrid_$CONFIG.py --validate-only --log-dir $logdir