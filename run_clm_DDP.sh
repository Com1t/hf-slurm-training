#!/bin/bash
#SBATCH --job-name=gpt2_ds_multi      ## job name
#SBATCH --nodes=2                ## request 2 nodes
#SBATCH --ntasks-per-node=1      ## run 1 srun task per node
#SBATCH --cpus-per-task=32       ## allocate 32 CPUs per srun task
#SBATCH --gres=gpu:8             ## request 8 GPUs per node
#SBATCH --time=00:10:00          ## run for a maximum of 10 minutes
#SBATCH --account="XXX"          ## PROJECT_ID, please fill in your project ID (e.g., XXX)
#SBATCH --partition=gp1d         ## gtest is for testing; you can change it to gp1d (1-day run), gp2d (2-day run), gp4d (4-day run), etc.
#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error output file

module purge
module load pkg/Anaconda3 cuda/11.7 compiler/gcc/11.2.0

conda activate alpaca

nvidia-smi

export GPUS_PER_NODE=8

# weights and bias API key
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=$(shuf -i 30000-60000 -n1)
export RDZV_ID=$(shuf -i 10-60000 -n1)

echo "NODELIST="${SLURM_JOB_NODELIST}
echo "MASTER_ADDR="${MASTER_ADDR}
echo "MASTER_PORT="${MASTER_PORT}

export OUTPUT_DIR=${HOME}/GPT_DDP_weights

srun bash -c \
  'torchrun --nnodes ${SLURM_NNODES} \
  --node_rank ${SLURM_PROCID} \
  --nproc_per_node ${GPUS_PER_NODE} \
  --rdzv_id ${RDZV_ID} \
  --rdzv_backend c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  ./run_clm.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_steps 250 \
  --learning_rate=5e-4 \
  --per_device_train_batch_size 8 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ${OUTPUT_DIR} \
  --report_to none \
  --do_train'
