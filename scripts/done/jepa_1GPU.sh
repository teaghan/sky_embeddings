#!/bin/bash
#SBATCH --nodes 1             # 1 node
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --account=rrg-kyi     # Priority account
#SBATCH --tasks-per-node=1    # 1 process per GPU.
#SBATCH --cpus-per-task=2     # 2 CPUs per GPU for dataloading.
#SBATCH --mem-per-cpu=16G     # 16GB of memory per CPU.
#SBATCH --time=0-03:00        # x hours # DD-HH:MM
#SBATCH --output=%N-%j.out    # Output file

module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5 hdf5/1.14.2 httpproxy
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate

export TORCH_NCCL_BLOCKING_WAIT=1  # Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=3456 # Store the master node’s port number in the MASTER_PORT environment variable.

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# env | grep SLURM
# env | grep CUDA

echo "$SLURM_NODEID master: $MASTER_ADDR"

echo "Copying data files to SLURM_TMPDIR.."

# Copy data files to local temporary directory
cp /project/rrg-kyi/astro/hsc/HSC_dud_galaxy_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_classifier_data_GIRYZ7610_64.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_regressor_data_GIRYZ7610_64.h5 $SLURM_TMPDIR

echo "$SLURM_NODEID: Launching python script"

# Execute the training script (tasks-per-node * nodes) times
srun python /home/heesters/projects/def-sfabbro/heesters/github/sky_embeddings/pretrain_mim.py jepa_1GPU --dist_backend nccl --init_method tcp://$MASTER_ADDR:$MASTER_PORT --world_size $SLURM_NTASKS  --batch_size 64 --verbose_iters 2000 --cp_time 10.00 --cp_freq 2500 -dd $SLURM_TMPDIR/