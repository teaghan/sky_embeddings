echo "Setting up interactive session.."
echo "Loading necessary environment modules.."

# Load required modules
module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5 hdf5/1.14.2 httpproxy

# Activate the Python environment
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(hostname)  # Capture and use the hostname as the master address
export MASTER_PORT=3456 # Store the master node’s port number in the MASTER_PORT environment variable.

# Echo the master address and node ID
echo "$SLURM_NODEID master: $MASTER_ADDR"

echo "Copying data files to local temporary directory.."

# Copy data files to local temporary directory
cp /project/rrg-kyi/astro/hsc/HSC_dud_galaxy_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_classifier_data_GIRYZ7610_64.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_regressor_data_GIRYZ7610_64.h5 $SLURM_TMPDIR

echo "Setup complete."

# # python /home/heesters/projects/def-sfabbro/heesters/github/sky_embeddings/pretrain_mim.py jepa_test --dist_backend nccl --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))  --batch_size 64 --verbose_iters 1 --cp_time 10.00 --cp_freq 100 -dd $SLURM_TMPDIR/