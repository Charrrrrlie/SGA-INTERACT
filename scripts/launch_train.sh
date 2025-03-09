#!/bin/bash

# Assign positional arguments to variables
partition=$1
num_gpu=$2
config_path=$3
extra_tag_info=$4

num_cpu=$((num_gpu * 6))
num_cpu=$((num_cpu > 48 ? 48 : num_cpu))

if [ ! -d "slurm_output" ]; then
    mkdir -p "slurm_output"
fi

# print the arguments
echo "partition: $partition num_gpu: $num_gpu num_cpu: $num_cpu"

OMP_NUM_THREADS=6 sbatch --gres=gpu:$num_gpu -n 1 --cpus-per-task=$num_cpu -p $partition -A $partition -o slurm_output/log.out.%j\
    train.sh $num_gpu $config_path $extra_tag_info