from logging import getLogger

import torch
import torch.distributed as dist

logger = getLogger()


def init_distributed(backend='nccl', init_method='env://', world_size=1, rank=0):
    """
    Initialize distributed training

    Args:
        backend (str, optional): The backend to use. Defaults to 'nccl', optimized for NVIDIA GPUs.
        init_method (str, optional): URL specifying how to initialize the process group. Defaults to 'env://'.
        world_size (int, optional): Number of available GPUs across all nodes. Defaults to 1.
        rank (int, optional): Unique process ID across all nodes. Defaults to 0.

    Returns:
        _type_: _description_
    """
    try:
        dist.init_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank
        )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'Distributed training not available {e}')
    return True


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
