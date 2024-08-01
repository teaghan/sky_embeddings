from contextlib import contextmanager
from logging import getLogger

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = getLogger()


@contextmanager
def distributed_env(backend, init_method, world_size, rank, local_rank, args=None):
    """
    Initialize distributed training environment

    Args:
        backend (str): backend to use
        init_method (str): initialization method
        world_size (int): number of available GPUs across all nodes
        rank (int): process ID across all nodes
        local_rank (int): process ID within a node
        args (any, optional): any additional arguments, not implemented. Defaults to None.
    """
    if world_size > 1:
        try:
            init_distributed(backend, init_method, world_size, rank, local_rank)
            yield
        finally:
            cleanup()
    else:
        logger.info('Distributed training not available, running on single GPU.')
        yield


def init_distributed(backend='nccl', init_method='env://', world_size=1, rank=0, local_rank=0):
    """
    Initialize distributed training

    Args:
        backend (str, optional): The backend to use. Defaults to 'nccl', optimized for NVIDIA GPUs.
        init_method (str, optional): URL specifying how to initialize the process group. Defaults to 'env://'.
        world_size (int, optional): Number of available GPUs across all nodes. Defaults to 1.
        rank (int, optional): Unique process ID across all nodes. Defaults to 0.

    Returns:
        bool: True if distributed training is available, False otherwise
    """
    try:
        logger.info(f'Rank {rank}/{world_size-1}: initializing process group..')
        dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'Distributed training not available {e}')

    logger.info(
        f'Rank {rank}/{world_size-1}: distributed training available: {dist.is_available()} and initialized: {dist.is_initialized()}'
    )
    logger.info(
        f'Rank {rank}/{world_size-1}: world size: {dist.get_world_size()}; rank: {dist.get_rank()}; local rank: {local_rank}'
    )

    return True


def cleanup():
    """
    Clean up distributed training
    """
    if dist.is_initialized():
        logger.info('Cleaning up distributed training..')
        dist.destroy_process_group()


def sync_barrier():
    """
    Synchronize all processes to wait for each other
    """
    if dist.is_initialized():
        dist.barrier()


def model_to_ddp(encoder, predictor, target_encoder, world_size, rank, current_device):
    # Distributed training
    logger.info(f'Rank {rank}/{world_size-1}: initializing models..')
    encoder = DDP(encoder, static_graph=False, device_ids=[current_device])
    predictor = DDP(predictor, static_graph=False, device_ids=[current_device])
    target_encoder = DDP(target_encoder, device_ids=[current_device])
    logger.info(f'Rank {rank}/{world_size-1}: models initialized.')
    return encoder, predictor, target_encoder


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
