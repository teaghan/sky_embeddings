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

# Echo the master address and node ID
echo "$SLURM_NODEID master: $MASTER_ADDR"

echo "Copying data files to local temporary directory.."

# Copy data files to local temporary directory
cp /project/rrg-kyi/astro/hsc/HSC_dud_galaxy_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_classifier_data_GIRYZ7610_64.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_regressor_data_GIRYZ7610_64.h5 $SLURM_TMPDIR

echo "Setup complete."

# salloc --nodes 1 --time=1:0:0 --tasks-per-node=1 --cpus-per-task=2 --mem-per-cpu=8G --account=def-sfabbro --gres=gpu:1 srun /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/jupyterlab.sh

# sshuttle --dns -Nr heesters@narval.computecanada.ca