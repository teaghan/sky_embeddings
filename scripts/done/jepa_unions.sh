#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=100G      
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5 hdf5/1.14.2 httpproxy
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Run JEPA training
srun python /home/heesters/projects/def-sfabbro/heesters/github/sky_embeddings/pretrain_jepa.py test --dist_backend nccl --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))  --batch_size 256 --verbose_iters 1 --cp_time 10.00
