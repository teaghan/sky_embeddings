import logging
import os

import torch

logger = logging.getLogger()


def cuda_setup(rank, local_rank, world_size):
    # Check for CUDA
    if torch.cuda.is_available():
        # Check number of visible GPUs for this process
        n_gpus_per_rank = torch.cuda.device_count()
        if rank == 0:
            logger.info(f'Using Torch version: {torch.__version__} with CUDA version {torch.version.cuda}.')  # type: ignore

        logger.info(f"Rank {rank}: CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f'Rank {rank}: number of GPUs visible: {torch.cuda.device_count()}')
        logger.info(f'Rank {rank}: local rank: {local_rank}: world size: {world_size}')

        current_device_id = torch.cuda.current_device()
        current_device = torch.device(f'cuda:{current_device_id}')

        # logger.info(f'Rank {rank}: CUDA Current Device ID: {torch.cuda.current_device()}')
        # logger.info(f'Rank {rank}: CUDA Current Device: {current_device}')
        # logger.info(f'Rank {rank}: CUDA Current Device Name: {torch.cuda.get_device_name(0)}')
        # logger.info(f'Rank {rank}: NCCL Version: {torch.cuda.nccl.version()}')

    else:
        # No GPU available, use CPU
        current_device = torch.device('cpu')
        current_device = 0
        if rank == 0:
            logger.info(f'CUDA is not available. Using Torch version: {torch.__version__} without CUDA.')
            logger.info(f'Using a {current_device} device.')

    return current_device, n_gpus_per_rank
