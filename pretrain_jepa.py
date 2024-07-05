import ast
import configparser
import copy
import logging
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from utils.dataloaders import (
    build_unions_dataloader,
)
from utils.datastream_utils import band_dict_incl
from utils.distributed import AllReduce, init_distributed
from utils.helper import setup_logging
from utils.jepa_masking import apply_masks, jepa_mask_generator
from utils.jepa_vit import build_model, build_optimizer, load_checkpoint
from utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger
from utils.misc import parseArguments

warnings.filterwarnings('ignore', category=UserWarning)

log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def main(args):
    # Set up directories
    cur_dir = os.path.dirname(__file__)
    config_dir = os.path.join(cur_dir, 'configs/')
    model_dir = os.path.join(cur_dir, 'models/')
    log_dir = os.path.join(cur_dir, 'logs/')
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(cur_dir, 'data/')
    fig_dir = os.path.join(cur_dir, 'figures/')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_dir + args.model_name + '.ini')

    # Initialize logging
    setup_logging(log_dir=log_dir, script_name=__file__, logging_level=config['LOGGING']['level'])
    logger = logging.getLogger()

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Check available GPUs on this node
        n_gpus_per_node = torch.cuda.device_count()
        # Local rank is the GPU ID within the node
        local_rank = int(os.environ.get('SLURM_LOCALID'))
        # Global rank is the GPU ID across all nodes
        rank = int(os.environ.get('SLURM_NODEID')) * n_gpus_per_node + local_rank
        # World size is the total number of GPUs across all nodes
        world_size = args.world_size
        # Set the current device
        current_device = local_rank
        torch.cuda.set_device(current_device)

        logger.info(f'Using Torch version: {torch.__version__} with CUDA {torch.version.cuda}.')
        logger.info(f'Using a {device} device with {n_gpus_per_node} GPU(s) per node.')

        # Initialize distributed training environment
        logger.info(f'From rank {rank}/{world_size-1}: initializing process group..')
        if init_distributed(
            backend=args.dist_backend,
            init_method=args.init_method,
            world_size=world_size,
            rank=rank,
        ):
            logger.info(f'From rank {rank}/{world_size-1}: process group initialized.')
            # Only display logs from rank 0 to avoid clutter
            if rank > 0:
                logger.setLevel(logging.ERROR)

    else:
        # No GPU available, use CPU
        device = torch.device('cpu')
        current_device = 0
        logger.info(
            f'CUDA is not available. Using Torch version: {torch.__version__} without CUDA.'
        )
        logger.info(f'Using a {device} device.')

    # Initialize torch multiprocessing
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # Initalize CSV logger
    csv_logger = CSVLogger(
        ('%d', 'iter'),
        ('%.5f', 'loss'),
        ('%.5f', 'mask-enc'),
        ('%.5f', 'mask-pred'),
        ('%d', 'time (ms)'),
        fname=os.path.join(log_dir, f'{args.model_name}_{rank}.csv'),
    )

    # Load model configuration
    model_name = args.model_name
    model_path = os.path.join(model_dir, model_name + '.pth.tar')
    latest_path = os.path.join(model_dir, model_name + '_latest.pth.tar')
    save_path = os.path.join(model_dir, model_name + '_iter_{current_iter}.pth.tar')

    # Display model configuration
    logger.info(f'Creating model: {model_name}')
    logger.info('Configuration:')
    for key_head in config.keys():
        if key_head == 'DEFAULT':
            continue
        logger.info(f'  {key_head}')
        for key in config[key_head].keys():
            logger.info(f'   {key}: {config[key_head][key]}')

    # Data parameters
    update_tiles = config['DATA']['update_tiles']
    min_num_bands = int(config['DATA']['min_bands'])
    processed_file = config['DATA']['processed_file']
    exclude_processed = config['DATA']['exclude_processed']
    obj_per_tile = int(config['DATA']['cutouts_per_tile'])
    crop_size = int(config['DATA']['img_size'])
    queue_length_datastream = int(config['DATA']['queue_length_datastream'])
    val_data_file = config['DATA']['val_data_file']
    off_limit_tiles = ast.literal_eval(config['DATA']['off_limit_tiles'])

    # Training parameters
    verbose_iters = args.verbose_iters
    cp_time = args.cp_time
    total_batch_iters = int(config['TRAINING']['total_batch_iters'])
    batch_size = int(config['TRAINING']['batch_size'])

    # Architecture parameters
    model_name = config['ARCHITECTURE']['model_type']
    num_channels = int(config['ARCHITECTURE']['num_channels'])
    patch_size = int(config['ARCHITECTURE']['patch_size'])
    pred_depth = int(config['ARCHITECTURE']['pred_depth'])
    pred_emb_dim = int(config['ARCHITECTURE']['pred_emb_dim'])
    use_bfloat16 = config['ARCHITECTURE']['use_bfloat16']
    do_norm = config['ARCHITECTURE']['normalization']
    pixel_mean = ast.literal_eval(config['ARCHITECTURE']['pixel_mean'])
    pixel_std = ast.literal_eval(config['ARCHITECTURE']['pixel_std'])

    # Masking parameters
    allow_overlap = config['MASK']['allow_overlap']  # overlap context/target blocks
    num_enc_masks = int(config['MASK']['num_enc_masks'])  # number of context blocks
    num_pred_masks = int(config['MASK']['num_pred_masks'])  # number of target blocks
    min_keep = int(config['MASK']['min_keep'])  # min number of patches in context block
    enc_mask_scale = ast.literal_eval(config['MASK']['enc_mask_scale'])  # scale of context blocks
    pred_mask_scale = ast.literal_eval(config['MASK']['pred_mask_scale'])  # scale of target blocks
    aspect_ratio_targets = config['MASK']['aspect_ratio_targets']  # ar of target blocks

    # Optimizer parameters
    ema = ast.literal_eval(config['OPTIMIZATION']['ema'])
    ipe_scale = float(config['OPTIMIZATION']['ipe_scale'])  # scheduler scale factor (def: 1.0)
    wd = float(config['OPTIMIZATION']['weight_decay'])
    final_wd = float(config['OPTIMIZATION']['final_weight_decay'])
    num_epochs = int(config['OPTIMIZATION']['epochs'])
    warmup = int(config['OPTIMIZATION']['warmup'])
    start_lr = float(config['OPTIMIZATION']['start_lr'])
    lr = float(config['OPTIMIZATION']['lr'])
    final_lr = float(config['OPTIMIZATION']['final_lr'])

    # Initialize model
    encoder, predictor = build_model(
        device=current_device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    # Inititialize mask generator
    mask_generator = jepa_mask_generator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio_targets,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    # Initialize data loader
    dataloader_train = build_unions_dataloader(
        update_tiles=update_tiles,
        off_limit_tiles=off_limit_tiles,
        min_num_bands=min_num_bands,
        processed_file=processed_file,
        exclude_processed=exclude_processed,
        obj_per_tile=obj_per_tile,
        queue_length=queue_length_datastream,
        batch_size=batch_size,
        in_dict=band_dict_incl,
        world_size=world_size,
        rank=rank,
        collator=mask_generator,
        patch_size=patch_size,
        num_channels=num_channels,
        eval=False,
        img_size=crop_size,
        eval_data_file=val_data_file,
        do_norm=do_norm,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    # Initialize optimizer
    optimizer, scaler, scheduler, wd_scheduler = build_optimizer(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=total_batch_iters,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    # Distributed training
    logger.info(f'From rank {rank}/{world_size-1}: initializing models..')
    encoder = DistributedDataParallel(encoder, static_graph=False, device_ids=[current_device])
    predictor = DistributedDataParallel(predictor, static_graph=False, device_ids=[current_device])
    target_encoder = DistributedDataParallel(target_encoder, device_ids=[current_device])
    logger.info(f'From rank {rank}/{world_size-1}: models initialized.')

    for p in target_encoder.parameters():
        p.requires_grad = False

    # Momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (total_batch_iters * num_epochs * ipe_scale)
        for i in range(int(total_batch_iters * num_epochs * ipe_scale) + 1)
    )

    if os.path.exists(model_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_iter = load_checkpoint(
            device=current_device,
            model_path=model_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_iter):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_generator.step()
    else:
        logger.info('No model checkpoint found. Starting from scratch.')
        start_iter = 0

    def save_checkpoint(cur_iter):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'cur_iter': cur_iter,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (cur_iter + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(current_iter=f'{cur_iter+1}'))
            if cur_iter == total_batch_iters:
                torch.save(save_dict, save_path.format(current_iter='final'))

    # Training loop
    logger.info('Starting training at batch iteration %d' % (start_iter))
    logger.info(
        f'Training the network with a batch size of {dataloader_train.batch_size} per GPU ..'
    )
    logger.info(
        f'Progress will be displayed every {verbose_iters} batch iterations and the model will be saved every {cp_time} minutes.'
    )

    loss_meter = AverageMeter()
    mask_enc_meter = AverageMeter()
    mask_pred_meter = AverageMeter()
    time_meter = AverageMeter()

    # Train the neural network
    losses_cp = defaultdict(list)
    logger('losses_cp:', losses_cp)
    cp_start_time = time.time()
    cur_iter = start_iter
    while cur_iter < total_batch_iters:
        for data, metadata, masks_enc, masks_pred in dataloader_train:
            logger('cur_iter:', cur_iter)

            def to_device():
                images = data.to(current_device, non_blocking=True)
                meta = metadata.to(current_device, non_blocking=True)
                masks_encoder = [mask.to(current_device, non_blocking=True) for mask in masks_enc]
                masks_predictor = [
                    mask.to(current_device, non_blocking=True) for mask in masks_pred
                ]
                return images, meta, masks_encoder, masks_predictor

            images, meta, masks_enc, masks_pred = to_device()
            mask_enc_meter.update(len(masks_enc[0][0]))
            mask_pred_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(images)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature dimension
                        B = len(h)
                        # create target representations
                        h = apply_masks(h, masks_pred)
                        h = torch.repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(images, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def compute_loss(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = compute_loss(z, h)

                # Backward pass + step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return float(loss), _new_lr, _new_wd, grad_stats

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(cur_iter, loss, mask_enc_meter.val, mask_pred_meter.val, etime)
                if (cur_iter % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[iter: %5d] loss: %.3f '
                        'masks: %.1f %.1f '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '(%.1f ms)'
                        % (
                            cur_iter,
                            loss_meter.avg,
                            mask_enc_meter.avg,
                            mask_pred_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            time_meter.avg,
                        )
                    )

                    if grad_stats is not None:
                        logger.info(
                            'iter: [%5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (
                                cur_iter,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

            # Increase the iteration
            cur_iter += 1

            # Save Checkpoint
            # after every cp_time minutes or
            # after every checkpoint_freq iterations or
            # at the end of training
            if (
                ((time.time() - cp_start_time) >= cp_time * 60)
                or (cur_iter + 1 % checkpoint_freq == 0)
                or (cur_iter == total_batch_iters)
            ):
                logger.info(
                    f'Saving checkpoint at iteration {cur_iter} after {cp_time} minutes of training.'
                )
                save_checkpoint(cur_iter)
                cp_start_time = time.time()

            logger.info('avg. loss %.3f' % loss_meter.avg)


# Run the training
if __name__ == '__main__':
    args = parseArguments()
    args = args.parse_args()
    main(args)

    logging.info('\nTraining complete.')
