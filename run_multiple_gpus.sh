# Requires: NUM_GPU

OMP_NUM_THREADS=8 WANDB_PROJECT=ptp torchrun --nnodes=1 --nproc_per_node=$NUM_GPU run.py $1
