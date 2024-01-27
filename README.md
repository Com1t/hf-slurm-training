# hf-slurm-training

**Table Of Contents**

- [Description](#description)
- [Environment](#environment)
- [Prerequisites](#prerequisites)
- [Slrum integration](#slurm-integration)
	- [Allocating resource](#allocating-resource)
	- [Environment preparation](#environment-preparation)
	- [Torchrun preparation](#torchrun-preparation)
	- [Srun and Torchrun](#srun-and-torchrun)
- [Running the sample](#running-the-sample)
	- [Language model training](#language-model-training)
	- [GPT-2 and causal language modeling](#gpt-2-and-causal-language-modeling)
	- [Streaming](#streaming)
	- [Low Cpu Memory Usage](#low-cpu-memory-usage)
 	- [ZeRO optimizer](#zero-optimizer)
	- [Creating a model on the fly](#creating-a-model-on-the-fly)

## Description

This repository demonstrates how to utilize the slurm scheduler for running the [language model training](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) from the Transformers library. 
While the original repository covered various training losses, this repository focuses on using slurm with the CLM loss for simplicity.

## Environment

1. Use `startDocker.sh` to create the torch environment.

2. (Optional) `startJupyterLabOnly.sh` can be used to create a Jupyter Lab environment.

## Prerequisites

1. Upgrade pip version and install the huggingface/transformers.
    ```bash
    pip3 install --upgrade pip
    pip3 install transformers
    ```

## Slrum integration

Slurm is a commonly used platform in HPC (High-Performance Computing). Most supercomputers use this scheduling system because it features excellent support for scientific computing, including MPI programs. However, Transformers utilizes Torch Distributed for encapsulation, a superset of MPI, so torchrun is needed. Using torchrun can include certain environment variables. While this repository mainly focuses on training, this process applies to all environments requiring torchrun.

This repository includes `run_clm_DDP.sh`, a script of multi-node torch training with slurm.
To run this SBATCH script, simply `sbatch run_clm_DDP.sh`.

### Allocating resource

In SBATCH scripts, it's necessary to specify the required resources, how to use them, billing account, and file output details at the beginning. In this example, Slurm allocates two nodes, each with 8 GPUs. Each node runs one SRUN task, and each task can utilize 32 CPUs. 
For the CPU allocation, TWCC supercomputer (it should be similar to other environments) follows the GPU-to-CPU ratio principle, which is typically 4 CPUs per GPU. Therefore, when the number of GPUs changes, the CPU count should also be adjusted accordingly.

```bash
#!/bin/bash
#SBATCH --job-name=gpt2_multi    ## job name
#SBATCH --nodes=2                ## request 2 nodes
#SBATCH --ntasks-per-node=1      ## run 1 srun task per node
#SBATCH --cpus-per-task=32       ## allocate 32 CPUs per srun task
#SBATCH --gres=gpu:8             ## request 8 GPUs per node
#SBATCH --time=00:10:00          ## run for a maximum of 10 minutes
#SBATCH --account="XXX"          ## PROJECT_ID, please fill in your project ID (e.g., XXX)
#SBATCH --partition=gp1d         ## gtest is for testing; you can change it to gp1d (1-day run), gp2d (2-day run), gp4d (4-day run), etc.
#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error output file
```

### Environment preparation

In the HPC environment, environment module is often used to handle the dependency. The following commands ensure the presence of Conda, CUDA, and GCC in the environment. I installed all python dependencies in the "gpt2" Conda environment, so it's activated by `conda activate gpt2`.

```bash
module purge
module load pkg/Anaconda3 cuda/11.7 compiler/gcc/11.2.0

conda activate gpt2
```

### Torchrun preparation

torchrun requires `MASTER_ADDR`, `MASTER_PORT`, `RDZV_ID` parameters. Here, I specify the first node from ${SLURM_JOB_NODELIST} as the master node, use a random port between 30000-60000 as the master port, and set RDZV_ID to a random number between 10-60000.
Additionally, I specify OUTPUT_DIR to where the results will be stored.

```bash
# weights and bias API key
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=$(shuf -i 30000-60000 -n1)
export RDZV_ID=$(shuf -i 10-60000 -n1)

echo "NODELIST="${SLURM_JOB_NODELIST}
echo "MASTER_ADDR="${MASTER_ADDR}
echo "MASTER_PORT="${MASTER_PORT}

export OUTPUT_DIR=${HOME}/GPT_DDP_weights
```

### Srun and Torchrun

After all the preparations, the final step is to use SRUN to execute the training on each node, running ${GPUS_PER_NODE} processes.
```bash
srun --jobid $SLURM_JOBID bash -c \
    'torchrun --nnodes ${SLURM_NNODES} \
    --node_rank ${SLURM_PROCID} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --rdzv_id ${RDZV_ID} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    ./run_clm.py \
    [...]' \
```


## Running the sample
### Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling (CLM) loss.

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.


The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for training and validation. We give examples of both below.

### GPT-2 and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before the tokenization). The loss here is that of causal language modeling.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_clm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm
```

### Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` to the command line. This is currently supported by `run_clm.py`.

### Low Cpu Memory Usage

To use low cpu memory mode which can be very useful for LLM, add `--low_cpu_mem_usage` to the command line. This is currently supported by `run_clm.py` and `run_clm_no_trainer.py`.

### ZeRO optimizer

[Installation](https://huggingface.co/docs/transformers/main_classes/deepspeed#installation)

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. If you wish to reduce the memory usage, the Zero Redundancy Optimizer (ZeRO) can eliminate redundant model copies stored among GPUs. Additionally, it can offload model weights to CPU memory, further reducing the memory requirements.

- DeepSpeed ZeRO-2 is primarily used only for training, as its features are of no use to inference.

- DeepSpeed ZeRO-3 can be used for inference as well, since it allows huge models to be loaded on multiple GPUs, which won’t be possible on a single GPU.

Using ZeRO with transformers model is quick and easy because all you need is to change a few configurations in the DeepSpeed configuration json. No code changes are needed.
You can find some examples of ZeRO configurations in the `configs` folder, including ZeRO-2 and ZeRO-3, as well as individual offload versions. If you want to learn more about how to adjust ZeRO configurations, you can refer to the [DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed) and [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/).

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --deepspeed "./configs/[].json" \
    [...]
```

### Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
    [...]
```

This feature is only available in `run_clm.py`.
