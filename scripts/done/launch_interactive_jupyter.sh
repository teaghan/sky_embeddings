echo "Setting up interactive session.."
echo "Loading necessary environment modules.."

# Load required modules
module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5 hdf5/1.14.2 httpproxy

# Activate the Python environment
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate

# salloc --nodes 1 --time=1:0:0 --tasks-per-node=1 --cpus-per-task=2 --mem-per-cpu=8G --account=def-sfabbro --gres=gpu:1 srun /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/jupyterlab.sh

# sshuttle --dns -Nr heesters@narval.computecanada.ca