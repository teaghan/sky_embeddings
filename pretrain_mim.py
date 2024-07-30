import ast
import atexit
import configparser
import copy
import logging
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
import time
import warnings
from collections import defaultdict

from utils.logging import setup_logging

setup_logging(
    log_dir='./logs',
    script_name=__file__,
    logging_level=logging.INFO,
)
logger = logging.getLogger()


from utils.cleanup import H5FileRegistry, SharedMemoryRegistry  # noqa: E402

# Register shared memory cleanup
atexit.register(SharedMemoryRegistry.cleanup_all)
atexit.register(H5FileRegistry.cleanup_all)


import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from utils.cuda_init import cuda_setup  # noqa: E402
from utils.dataloaders import build_fits_dataloader, build_h5_dataloader  # noqa: E402
from utils.distributed import AllReduce, distributed_env  # noqa: E402
from utils.eval_fns import mae_predict  # noqa: E402
from utils.jepa_masking import apply_masks  # noqa: E402
from utils.jepa_masking import jepa_mask_generator as jepa_mask_gen  # noqa: E402
from utils.jepa_schedulers import build_momentum_scheduler  # noqa: E402
from utils.jepa_tensors import repeat_interleave_batch  # noqa: E402
from utils.jepa_vit import get_num_patches  # noqa: E402
from utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger  # noqa: E402
from utils.mim_vit import build_model  # noqa: E402
from utils.misc import parseArguments  # noqa: E402
from utils.plotting_fns import plot_batch, plot_progress, visualize_masks  # noqa: E402
from utils.pretrain_fns import (  # noqa: E402
    distributed_linear_probe,
    linear_probe,
    log_current_status,
    run_iter,
    val_iter,
)

warnings.filterwarnings('ignore', category=UserWarning)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def main(args):
    # Directories
    cur_dir = os.path.dirname(__file__)
    config_dir = os.path.join(cur_dir, 'configs')
    model_dir = os.path.join(cur_dir, 'models')
    log_dir = os.path.join(cur_dir, 'logs')
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(cur_dir, 'data')
    fig_dir = os.path.join(cur_dir, 'figures')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load model configuration
    model_name = args.model_name
    config = configparser.ConfigParser()
    config.read(os.path.join(config_dir, model_name + '.ini'))

    logger.debug(f'Loading model configuration from {os.path.join(config_dir, model_name+".ini")}')

    # Rank is the GPU ID across all nodes
    rank = int(os.environ['SLURM_PROCID'])
    # Local rank is the GPU ID within the node
    local_rank = int(os.environ.get('SLURM_LOCALID'))  # type: ignore
    # World size is the total number of GPUs across all nodes
    world_size = int(os.environ.get('SLURM_NTASKS'))  # type: ignore

    with distributed_env(
        backend=args.dist_backend,
        init_method=args.init_method,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    ):
        current_device, n_gpus_per_rank = cuda_setup(rank, local_rank, world_size)

        # Only display logs from rank 0 to avoid clutter
        # if rank > 0:
        #     logger.setLevel(logging.ERROR)

        # Initalize CSV logger
        csv_logger = CSVLogger(
            ('%d', 'iter'),
            ('%.5f', 'loss'),
            ('%.5f', 'mask-enc'),
            ('%.5f', 'mask-pred'),
            ('%d', 'time (ms)'),
            fname=os.path.join(log_dir, f'{args.model_name}_{rank}.csv'),
            mode='overwrite',
        )

        # Data parameters
        min_num_bands = int(config['DATA']['min_bands'])
        bands = ast.literal_eval(config['DATA']['bands'])
        obj_per_tile = int(config['DATA']['cutouts_per_tile'])
        crop_size = int(config['DATA']['img_size'])
        val_data_file = os.path.join(data_dir, config['DATA']['val_data_file'])
        lp_class_data_file = (
            os.path.join(data_dir, config['DATA']['lp_class_data_file'])
            if 'lp_class_data_file' in config['DATA']
            else None
        )
        lp_regress_data_file = (
            os.path.join(data_dir, config['DATA']['lp_regress_data_file'])
            if 'lp_regress_data_file' in config['DATA']
            else None
        )
        use_calexp = ast.literal_eval(config['DATA']['use_calexp'])
        lp_combine = config['DATA']['lp_combine']
        val_batches = int(config['DATA']['val_batches'])

        # Training parameters
        verbose_iters = args.verbose_iters
        cp_time = args.cp_time
        cp_freq = args.cp_freq
        total_batch_iters = int(ast.literal_eval(config['TRAINING']['total_batch_iters']))
        batch_size = int(config['TRAINING']['batch_size'])
        ema = ast.literal_eval(config['TRAINING']['ema'])
        use_bfloat16 = ast.literal_eval(config['TRAINING']['use_bfloat16'])

        # Architecture parameters
        model_type = config['ARCHITECTURE']['model_type']
        num_channels = int(config['ARCHITECTURE']['num_channels'])
        patch_size = int(config['ARCHITECTURE']['patch_size'])

        # Masking parameters
        max_mask_ratio = float(config['MASK']['max_mask_ratio'])
        mask_ratio = float(config['MASK']['mask_ratio'])
        allow_overlap = ast.literal_eval(config['MASK']['allow_overlap'])  # overlap context/target blocks
        num_enc_masks = int(config['MASK']['num_enc_masks'])  # number of context blocks
        num_pred_masks = int(config['MASK']['num_pred_masks'])  # number of target blocks
        min_keep = int(config['MASK']['min_keep'])  # min number of patches in context block
        enc_mask_scale = ast.literal_eval(config['MASK']['enc_mask_scale'])  # scale of context blocks
        pred_mask_scale = ast.literal_eval(config['MASK']['pred_mask_scale'])  # scale of target blocks
        aspect_ratio_targets = ast.literal_eval(config['MASK']['aspect_ratio_targets'])  # ar of target blocks

        # Display model configuration
        if rank == 0:
            logger.info(f'Creating model: {model_name}')
            logger.info('Configuration:')
            for key_head in config.keys():
                if key_head == 'DEFAULT':
                    continue
                logger.info(f'{key_head}')
                for key in config[key_head].keys():
                    logger.info(f'  {key}: {config[key_head][key]}')

        # Construct the model and optimizer
        model_path = os.path.join(model_dir, model_name + '.pth.tar')
        latest_path = os.path.join(model_dir, model_name + '_latest.pth.tar')
        save_path = os.path.join(model_dir, model_name + '_iter_{current_iter}.pth.tar')

        model, losses, cur_iter, optimizer, lr_scheduler, wd_scheduler, scaler = build_model(  # type: ignore
            config, model_path, current_device, build_optimizer=True
        )

        if 'jepa' in model_type and isinstance(model, tuple):
            # Inititialize mask generator
            mask_generator = jepa_mask_gen(
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
            collator = mask_generator

            encoder, predictor = model
            target_encoder = copy.deepcopy(encoder)

            # Distributed training
            logger.info(f'Rank {rank}/{world_size-1}: initializing models..')
            encoder = DDP(encoder, static_graph=False, device_ids=[current_device])
            predictor = DDP(predictor, static_graph=False, device_ids=[current_device])
            target_encoder = DDP(target_encoder, device_ids=[current_device])
            logger.info(f'Rank {rank}/{world_size-1}: models initialized.')

            dist.barrier()
            logger.debug(f'Rank {rank}/{world_size-1} passed the synchronization barrier.')

            # Target encoder is not trained through backprop
            for p in target_encoder.parameters():
                p.requires_grad = False

            # Momentum scheduler to update the target encoder
            momentum_scheduler = build_momentum_scheduler(ema[0], ema[1], total_batch_iters)
            # Get the scheduler to the correct iteration
            for _ in range(cur_iter):
                next(momentum_scheduler)
        else:
            collator = None

        # Build dataloaders
        # num_workers = min([os.cpu_count(), 12 * n_gpus_per_node])  # type: ignore
        # logger.info(f'Using {os.cpu_count()} cpus for data loading. Num workers is set to {num_workers}.')
        # if num_workers > 1:
        #     num_workers -= 1
        num_workers = 2 * n_gpus_per_rank

        if 'train_data_file' in config['DATA']:
            # Using .h5 training file
            dataloader_train = build_h5_dataloader(
                os.path.join(data_dir, config['DATA']['train_data_file']),
                batch_size=batch_size,
                bands=bands,
                num_workers=num_workers,
                patch_size=patch_size,
                num_channels=num_channels,
                max_mask_ratio=max_mask_ratio,
                img_size=crop_size,
                num_patches=get_num_patches(model),
                shuffle=True,
            )
            logger.info(f'The training set consists of {len(dataloader_train.dataset)} cutouts.')
            train_nested_batches = False
        else:
            # Using fits files in training directory
            # Might need to decrease num_workers and increase cutouts_per_tile
            _, dataloader_train, datasampler_train = build_fits_dataloader(
                ast.literal_eval(config['DATA']['train_data_paths']),
                bands=bands,
                min_bands=min_num_bands,
                batch_size=batch_size,
                num_workers=num_workers,
                patch_size=patch_size,
                max_mask_ratio=max_mask_ratio,
                img_size=crop_size,
                cutouts_per_tile=obj_per_tile,
                use_calexp=use_calexp,
                model_type=model_type,
                collator=collator,
                world_size=world_size,
                rank=rank,
                current_device=current_device,  # type: ignore
                ra_dec=True,
                augment=False,
                shuffle=True,
            )
            train_nested_batches = True

            logger.info(
                f'Rank {rank}: initialized train dataloader with {len(dataloader_train.dataset)} cutouts.'
            )

        dataloader_val = build_h5_dataloader(
            val_data_file,
            batch_size=batch_size,
            bands=bands,
            num_workers=num_workers,
            patch_size=patch_size,
            num_channels=num_channels,
            max_mask_ratio=max_mask_ratio,
            img_size=crop_size,
            num_patches=get_num_patches(model),
            shuffle=True,
            collator=mask_generator,
            num_batches=val_batches // world_size,
            model_type=model_type,
            seed=_GLOBAL_SEED,
            world_size=world_size,
            rank=rank,
        )

        logger.info(
            f'Rank {rank}: initialized validation dataloader with {len(dataloader_val.dataset)} cutouts.'
        )

        # Training loop
        if 'jepa' in model_type and isinstance(model, tuple):
            train_network_jepa(
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                dataloader_train=dataloader_train,
                dataloader_val=dataloader_val,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                momentum_scheduler=momentum_scheduler,
                scaler=scaler,
                losses=losses,
                cur_iter=cur_iter,
                total_batch_iters=total_batch_iters,
                verbose_iters=verbose_iters,
                cp_time=cp_time,
                cp_freq=cp_freq,
                model_name=model_name,
                fig_dir=fig_dir,
                latest_path=latest_path,
                save_path=save_path,
                world_size=world_size,
                rank=rank,
                current_device=current_device,
                logger=logger,
                csv_logger=csv_logger,
                use_bfloat16=use_bfloat16,
                lp_class_data_file=lp_class_data_file,
                lp_regress_data_file=lp_regress_data_file,
                lp_combine=lp_combine,
                patch_size=patch_size,
                bands=bands,
            )

        else:
            train_network(
                model,
                dataloader_train,
                dataloader_val,
                train_nested_batches,
                optimizer,
                lr_scheduler,
                current_device,
                mask_ratio,
                losses,
                cur_iter,
                total_batch_iters,
                args.verbose_iters,
                args.cp_time,
                model_path,
                fig_dir,
                lp_class_data_file,
                lp_regress_data_file,
                lp_combine,
                logger,
            )


def get_train_samples(dataloader, train_nested_batches):
    """Accomodates both dataloaders."""
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks, ra_decs in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask, ra_dec in zip(sample_batches[0], masks[0], ra_decs[0]):
                yield samples, mask, ra_dec
    else:
        for samples, mask, ra_dec in dataloader:
            yield samples, mask, ra_dec


def train_network_jepa(
    encoder,
    predictor,
    target_encoder,
    dataloader_train,
    dataloader_val,
    optimizer,
    lr_scheduler,
    wd_scheduler,
    momentum_scheduler,
    scaler,
    losses,
    cur_iter,
    total_batch_iters,
    verbose_iters,
    cp_time,
    cp_freq,
    model_name,
    fig_dir,
    latest_path,
    save_path,
    world_size,
    rank,
    current_device,
    logger,
    csv_logger,
    use_bfloat16,
    lp_class_data_file,
    lp_regress_data_file,
    lp_combine,
    patch_size,
    bands,
):
    if rank == 0:
        logger.info(f'Training the network with a batch size of {dataloader_train.batch_size} per GPU ...')
        logger.info(
            f'Progress will be displayed every {verbose_iters} batch iterations and the model will be saved every {cp_time} minutes.'
        )

    loss_meter = AverageMeter()
    mask_enc_meter = AverageMeter()
    mask_pred_meter = AverageMeter()
    time_meter = AverageMeter()

    def save_checkpoint(cur_iter):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'wd_scheduler': None if wd_scheduler is None else wd_scheduler.state_dict(),
            'lr': _new_lr,
            'cur_iter': cur_iter,
            'loss': loss_meter.avg,
            'world_size': world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (cur_iter + 1) % cp_freq == 0:
                torch.save(save_dict, save_path.format(current_iter=f'{cur_iter+1}'))
            if cur_iter == total_batch_iters:
                torch.save(save_dict, save_path.format(current_iter='final'))

    # Train the neural networks
    cp_start_time = time.time()
    losses_cp = defaultdict(list)
    if rank == 0:
        logger.info('Starting training loop...')
    while cur_iter < (total_batch_iters):
        # Iterate through training dataset
        if cur_iter % 250 == 0:
            logger.debug(f'Rank {rank}: iter: {cur_iter}')
        for data, metadata, masks_enc, masks_pred in dataloader_train:

            def to_device(data, metadata, masks_enc, masks_pred):
                images = data.to(current_device, non_blocking=True)
                meta = metadata.to(current_device, non_blocking=True)
                masks_encoder = [mask.to(current_device, non_blocking=True) for mask in masks_enc]
                masks_predictor = [mask.to(current_device, non_blocking=True) for mask in masks_pred]
                return images, meta, masks_encoder, masks_predictor

            images, meta, masks_enc, masks_pred = to_device(data, metadata, masks_enc, masks_pred)
            mask_enc_meter.update(len(masks_enc[0][0]))
            mask_pred_meter.update(len(masks_pred[0][0]))

            def train_step():
                encoder.train()
                predictor.train()
                target_encoder.eval()  # target encoder is not trained through backprop

                _new_lr = lr_scheduler.step()
                _new_wd = wd_scheduler.step()

                if cur_iter % 1000 == 0:
                    logger.info(
                        f'Rank {rank}: Iter: {cur_iter}; learning rate: {_new_lr:.5f}; wd: {_new_wd:.5f}'
                    )

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(images)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature dimension
                        B = len(h)
                        # create target representations
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))  # type: ignore
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
                    loss.backward()  # type: ignore
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return loss, _new_lr, _new_wd, grad_stats  # type: ignore

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)

            losses_cp['train_loss'].append(loss.item())  # type: ignore
            loss = float(loss)  # type: ignore
            loss_meter.update(loss)
            time_meter.update(etime)

            def log_stats(losses_cp):
                csv_logger.log(cur_iter, loss, mask_enc_meter.val, mask_pred_meter.val, etime)
                if rank == 0 and ((cur_iter % verbose_iters == 0) or np.isnan(loss) or np.isinf(loss)):  # type: ignore
                    logger.info(
                        f'[iter: {cur_iter:5d}] [loss: {loss_meter.avg:.3f}] '
                        f'[masks: {mask_enc_meter.avg:.1f}, {mask_pred_meter.avg:.1f}] '
                        f'[wd: {_new_wd:.2e}] [lr: {_new_lr:.2e}] '
                        f'[mem: {torch.cuda.max_memory_allocated() / 1024.0**2:.2e}] '
                        f'({time_meter.avg:.1f} ms)'
                    )

                    # This might be redundant, leaving here for now
                    for k in losses_cp.keys():
                        losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                    losses['batch_iters'].append(cur_iter)

                    log_current_status(
                        cur_iter, total_batch_iters, losses, lp_class_data_file, lp_regress_data_file
                    )

                    # Reset checkpoint loss dictionary
                    losses_cp = defaultdict(list)

                    if grad_stats is not None:
                        logger.info(
                            'iter: [%5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (
                                cur_iter,
                                grad_stats.first_layer,  # type: ignore
                                grad_stats.last_layer,  # type: ignore
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            def validate():
                encoder.eval()
                predictor.eval()
                target_encoder.eval()
                with torch.no_grad():
                    images_plot = []
                    meta_plot = []
                    masks_enc_plot = []
                    masks_pred_plot = []
                    plot_idx = random.randint(0, len(dataloader_val) - 1)
                    for i, (data, metadata, masks_enc, masks_pred) in enumerate(dataloader_val):
                        images, meta, masks_encoder, masks_predictor = to_device(
                            data, metadata, masks_enc, masks_pred
                        )

                        if i == plot_idx:
                            images_plot.append(images)
                            meta_plot.append(meta)
                            masks_enc_plot.append(masks_encoder)
                            masks_pred_plot.append(masks_predictor)

                        loss = val_iter(
                            encoder,
                            predictor,
                            target_encoder,
                            images,
                            masks_encoder,
                            masks_predictor,
                            use_bfloat16,
                        )
                        losses_cp['val_loss'].append(loss.item())  # type: ignore

                    logger.debug(
                        f'Rank {rank}: Validation done at iteration {cur_iter}. Continuing with classifier and regressor.'
                    )

                    dist.barrier()
                    logger.debug(f'Rank {rank}: passed barrier after initial validation.')

                    if lp_class_data_file or lp_regress_data_file:
                        # Run Linear Probing tests
                        distributed_linear_probe(
                            model=target_encoder,
                            losses_cp=losses_cp,
                            device=current_device,
                            dataloader_template=dataloader_val,
                            model_type='jepa',
                            class_data_path=lp_class_data_file,
                            regress_data_path=lp_regress_data_file,
                            combine=lp_combine,
                            world_size=world_size,
                            rank=rank,
                        )
                    if rank == 0:
                        if len(losses['batch_iters']) > 1:
                            # Plot progress
                            plot_progress(
                                losses,
                                y_lims=[(0, 0.7), (0.8, 1.0), (0.6, 1.0)],
                                savename=os.path.join(fig_dir, f'{model_name}_progress.png'),
                            )
                            # Plot 5 sample masks
                            images_plot = images_plot[0].de
                            visualize_masks(
                                images=images_plot,
                                masks_enc=masks_enc_plot,
                                masks_pred=masks_pred_plot,
                                patch_size=patch_size,
                                bands=bands,
                                savename=os.path.join(fig_dir, f'{model_name}_mask_examples.png'),
                            )

            # Perform validation every 5000 iterations
            if cur_iter % verbose_iters == 0:
                if dist.is_initialized():
                    dist.barrier()
                    logger.info(f'Rank {rank}: performing validation at iteration {cur_iter}')
                    try:
                        validate()
                        logger.info(
                            f'Rank {rank}: Validation done at iteration {cur_iter}. Pretraining continues.'
                        )
                    except Exception as e:
                        logger.error(f'Rank {rank}: Validation failed at iteration {cur_iter}: {e}')

                    # Synchronize all processes after validation
                    dist.barrier()
                    logger.debug(f'Rank {rank}: passed barrier after final validation step.')
                else:
                    logger.info(f'Rank {rank}: distributed environment not initialized at validation.')
                    logger.info(f'Rank {rank}: performing validation at iteration {cur_iter}')
                    try:
                        validate()
                        logger.info(
                            f'Rank {rank}: Validation done at iteration {cur_iter}. Continuing training.'
                        )
                    except Exception as e:
                        logger.error(f'Rank {rank}: Validation failed at iteration {cur_iter}: {e}')

            log_stats(losses_cp)

            assert not np.isnan(loss), 'loss is nan'
            assert not np.isinf(loss), 'loss is inf'

            # Increase the iteration
            cur_iter += 1

            # Save Checkpoint
            # after every cp_time minutes or
            # after every checkpoint_freq iterations or
            # at the end of training
            cur_time = time.time() - cp_start_time
            if (cur_time >= cp_time * 60) or (cur_iter % cp_freq == 0) or (cur_iter == total_batch_iters):
                logger.info(
                    f'Saving checkpoint at iteration {cur_iter} after {cur_time} minutes of training.'
                )
                save_checkpoint(cur_iter)
                cp_start_time = time.time()

            if rank == 0 and cur_iter % 100 == 0:
                logger.info(f'Iter: {cur_iter}: avg. loss {loss_meter.avg:3f}')


def train_network(
    model,
    dataloader_train,
    dataloader_val,
    train_nested_batches,
    optimizer,
    lr_scheduler,
    current_device,
    mask_ratio,
    losses,
    cur_iter,
    total_batch_iters,
    verbose_iters,
    cp_time,
    model_filename,
    fig_dir,
    lp_class_data_file,
    lp_regress_data_file,
    lp_combine,
    logger,
):
    logger.info(f'Training the network with a batch size of {dataloader_train.batch_size} per GPU ...')
    logger.info(
        f'Progress will be displayed every {verbose_iters} batch iterations and the model will be saved every {cp_time} minutes.'
    )

    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    # time1 = time.time()
    while cur_iter < (total_batch_iters):
        # Iterate through training dataset
        for samples, masks, ra_decs in get_train_samples(dataloader_train, train_nested_batches):
            # Switch to GPU if available
            samples = samples.to(current_device, non_blocking=True)
            masks = masks.to(current_device, non_blocking=True)
            ra_decs = ra_decs.to(current_device, non_blocking=True)

            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(
                model,
                samples,
                ra_decs,
                masks,
                mask_ratio,
                optimizer,
                lr_scheduler,
                losses_cp,
                mode='train',
            )

            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:
                with torch.no_grad():
                    # Calculate average loss on validation set
                    for i, (samples, masks, ra_decs) in enumerate(dataloader_val):
                        # Switch to GPU if available
                        samples = samples.to(current_device, non_blocking=True)
                        masks = masks.to(current_device, non_blocking=True)
                        ra_decs = ra_decs.to(current_device, non_blocking=True)

                        # Run an iteration
                        model, optimizer, lr_scheduler, losses_cp = run_iter(
                            model,
                            samples,
                            ra_decs,
                            masks,
                            mask_ratio,
                            optimizer,
                            lr_scheduler,
                            losses_cp,
                            mode='val',
                        )
                        # Don't bother with the whole dataset
                        if i >= 200:
                            break

                    if lp_class_data_file or lp_regress_data_file:
                        # Run Linear Probing tests
                        linear_probe(
                            model,
                            losses_cp,
                            current_device,
                            dataloader_val,
                            lp_class_data_file,
                            lp_regress_data_file,
                            combine=lp_combine,
                        )

                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)

                # Print current status
                logger.info('Batch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                logger.info('Losses:')
                logger.info('\tTraining Dataset')
                logger.info('\t\tTotal Loss: %0.3f' % (losses['train_loss'][-1]))
                logger.info('\tValidation Dataset')
                logger.info('\t\tTotal Loss: %0.3f' % (losses['val_loss'][-1]))
                if lp_class_data_file or lp_regress_data_file:
                    logger.info('Linear Probing Results:')
                    if lp_class_data_file:
                        logger.info('\tClassification Accuracy:')
                        logger.info(
                            '\t\tTraining: %0.3f, Validation: %0.3f'
                            % (losses['train_lp_acc'][-1], losses['val_lp_acc'][-1])
                        )
                    if lp_regress_data_file:
                        logger.info('\tRegression R2')
                        logger.info(
                            '\t\tTraining: %0.3f, Validation: %0.3f'
                            % (losses['train_lp_r2'][-1], losses['val_lp_r2'][-1])
                        )

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)

                if len(losses['batch_iters']) > 1:
                    # Plot progress
                    plot_progress(
                        losses,
                        y_lims=[(0, 0.7), (0.8, 1.0), (0.6, 1.0)],
                        savename=os.path.join(
                            fig_dir, f'{os.path.basename(model_filename).split(".")[0]}_progress.png'
                        ),
                    )
                # Plot 5 validation samples
                pred_imgs, mask_imgs, orig_imgs = mae_predict(
                    model, dataloader_val, current_device, mask_ratio, single_batch=True
                )
                plot_batch(
                    orig_imgs,
                    mask_imgs,
                    pred_imgs,
                    n_samples=5,
                    channel_index=0,
                    savename=os.path.join(
                        fig_dir, f'{os.path.basename(model_filename).split(".")[0]}_{cur_iter}iters.png'
                    ),
                )

            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time * 60:
                # Save periodically
                logger.info('Saving network...')
                torch.save(
                    {
                        'batch_iters': cur_iter,
                        'losses': losses,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'model': model.module.state_dict(),
                    },
                    model_filename,
                )

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                # Save after training
                logger.info('Saving network...')
                torch.save(
                    {
                        'batch_iters': cur_iter,
                        'losses': losses,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'model': model.module.state_dict(),
                    },
                    model_filename,
                )
                # Finish training
                break


# Run the training
if __name__ == '__main__':
    args = parseArguments()
    args = args.parse_args()
    main(args)

    print('\nTraining complete.')
