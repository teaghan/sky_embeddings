import logging
import os

import torch

logger = logging.getLogger(__name__)


def setup_logging(log_dir, script_name, logging_level):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        script_name (str): script name
    """
    log_filename = os.path.join(log_dir, f'{os.path.splitext(os.path.basename(script_name))[0]}.log')

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
    )


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  # type: ignore

    result = closure()

    if log_timings:
        end.record()  # type: ignore
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):
    def __init__(self, *argv, fname='./logs/default.log'):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None  # type: ignore
    stats.last_layer = None  # type: ignore
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm  # type: ignore
                if stats.first_layer is None:  # type: ignore
                    stats.first_layer = grad_norm  # type: ignore
    if stats.first_layer is None or stats.last_layer is None:  # type: ignore
        stats.first_layer = stats.last_layer = 0.0  # type: ignore
    return stats


def log_current_status(
    cur_iter, total_batch_iters, losses, lp_class_data_file=None, lp_regress_data_file=None
):
    # Print current status
    logger.info(f'Batch Iterations: {cur_iter}/{total_batch_iters}')
    logger.info('Losses:')
    logger.info('Training Dataset')
    logger.info(f'  Total Loss: {losses["train_loss"][-1]:.3f}')
    logger.info('  Validation Dataset')
    logger.info(f'  Total Loss: {losses["val_loss"][-1]:.3f}')
    if lp_class_data_file is not None or lp_regress_data_file is not None:
        logger.info('Linear Probing Results:')
        if lp_class_data_file:
            logger.info('Classification Accuracy:')
            logger.info(
                f'  Training: {losses["train_lp_acc"][-1]:.3f}, Validation: {losses["val_lp_acc"][-1]:.3f}'
            )
        if lp_regress_data_file:
            logger.info('Regression R2')
            logger.info(
                f' Training: {losses["train_lp_r2"][-1]:.3f}, Validation: {losses["val_lp_r2"][-1]:.3f}'
            )
