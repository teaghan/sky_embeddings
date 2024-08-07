import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch
import warnings
from tqdm import tqdm  # Import tqdm for progress tracking
import logging  # Import logging for detailed logging

import cProfile
import pstats

warnings.filterwarnings("ignore", category=UserWarning)

from utils.misc import str2bool, parseArguments
from utils.pretrain_fns import run_iter, linear_probe
from utils.mim_vit import build_model, consistency_loss
from utils.dataloaders import build_h5_dataloader, build_fits_dataloader
from utils.plotting_fns import plot_progress, plot_batch
from utils.eval_fns import mae_predict
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_gpu = torch.cuda.device_count()

    logger.info(f'Using Torch version: {torch.__version__}')
    logger.info(f'Using a {device} device with {n_gpu} GPU(s)')

    # Directories
    cur_dir = os.path.dirname(__file__)
    config_dir = os.path.join(cur_dir, 'configs/')
    model_dir = os.path.join(cur_dir, 'models/')
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(cur_dir, 'data/')
    fig_dir = os.path.join(cur_dir, 'figures/')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Load model configuration
    model_name = args.model_name
    config = configparser.ConfigParser()
    config.read(config_dir + model_name + '.ini')
    
    
    # Get the resume parameter from the config
    resume = config['TRAINING'].getboolean('resume')

    # Display model configuration
    logger.info(f'\nCreating model: {model_name}')
    logger.info('\nConfiguration:')
    for key_head in config.keys():
        if key_head == 'DEFAULT':
            continue
        logger.info(f'  {key_head}')
        for key in config[key_head].keys():
            logger.info(f'    {key}: {config[key_head][key]}')

    # Define total_epochs
    total_epochs = int(config['TRAINING']['total_epochs'])  # Get total epochs from config

    # Construct the model, optimizer, etc.
    model_filename = os.path.join(model_dir, model_name + '.pth.tar')
    student_model, teacher_model, losses, cur_epoch, optimizer, lr_scheduler = build_model(config, model_filename, device, build_optimizer=True, resume=resume)
    
    # Data loader stuff
    num_workers = min([os.cpu_count(), 12 * n_gpu])
    if num_workers > 1:
        num_workers -= 1

    # Masking stuff
    if 'mim' in config['ARCHITECTURE']['model_type']:
        mask_ratio = None
        max_mask_ratio = float(config['TRAINING']['max_mask_ratio'])
    else:
        mask_ratio = float(config['TRAINING']['mask_ratio'])
        max_mask_ratio = None

    # Build dataloaders
    if 'train_data_file' in config['DATA']:
        # Using .h5 training file
        dataloader_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']),
                                               batch_size=int(config['TRAINING']['batch_size']),
                                               num_workers=num_workers,
                                               patch_size=int(config['ARCHITECTURE']['patch_size']),
                                               num_channels=int(config['ARCHITECTURE']['num_channels']),
                                               max_mask_ratio=max_mask_ratio,
                                               img_size=int(config['ARCHITECTURE']['img_size']),
                                               num_patches=student_model.module.patch_embed.num_patches,
                                               shuffle=True)
        logger.info(f'The training set consists of {len(dataloader_train.dataset)} cutouts.')
        train_nested_batches = False
    else:
        # Using fits files in training directory
        dataloader_train = build_fits_dataloader(eval(config['DATA']['train_data_paths']),
                                                 bands=eval(config['DATA']['bands']),
                                                 min_bands=int(config['DATA']['min_bands']),
                                                 batch_size=int(config['TRAINING']['batch_size']),
                                                 num_workers=num_workers,
                                                 patch_size=int(config['ARCHITECTURE']['patch_size']),
                                                 max_mask_ratio=max_mask_ratio,
                                                 img_size=int(config['ARCHITECTURE']['img_size']),
                                                 cutouts_per_tile=int(config['DATA']['cutouts_per_tile']),
                                                 use_calexp=str2bool(config['DATA']['use_calexp']),
                                                 ra_dec=True,
                                                 augment=False,
                                                 shuffle=True)
        train_nested_batches = True

    dataloader_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']),
                                         batch_size=int(config['TRAINING']['batch_size']),
                                         num_workers=num_workers,
                                         patch_size=int(config['ARCHITECTURE']['patch_size']),
                                         num_channels=int(config['ARCHITECTURE']['num_channels']),
                                         max_mask_ratio=max_mask_ratio,
                                         img_size=int(config['ARCHITECTURE']['img_size']),
                                         num_patches=student_model.module.patch_embed.num_patches,
                                         shuffle=True)

    # Linear probing validation data files
    lp_class_data_file = os.path.join(data_dir, config['DATA']['lp_class_data_file']) if 'lp_class_data_file' in config['DATA'] else None
    lp_regress_data_file = os.path.join(data_dir, config['DATA']['lp_regress_data_file']) if 'lp_regress_data_file' in config['DATA'] else None

    train_network(student_model, teacher_model, dataloader_train, dataloader_val, train_nested_batches,
                  optimizer, lr_scheduler, device,
                  mask_ratio,
                  losses, cur_epoch,
                  total_epochs, args.verbose_iters, args.cp_time, model_filename, fig_dir,
                  lp_class_data_file, lp_regress_data_file, config['DATA']['lp_combine'])

def get_train_samples(dataloader, train_nested_batches):
    '''Accommodates both dataloaders.'''
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks, ra_decs in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask, ra_dec in zip(sample_batches[0], masks[0], ra_decs[0]):
                yield samples, mask, ra_dec
    else:
        for samples, mask, ra_dec in dataloader:
            yield samples, mask, ra_dec
            
            
import matplotlib.pyplot as plt   
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']        
# Disable LaTeX rendering in Matplotlib
plt.rcParams['text.usetex'] = False

def train_network(student_model, teacher_model, dataloader_train, dataloader_val, train_nested_batches, optimizer, lr_scheduler, device, mask_ratio,
                  losses, cur_epoch, total_epochs, verbose_iters, cp_time, model_filename, fig_dir,
                  lp_class_data_file, lp_regress_data_file, lp_combine):
    logger.info(f'Training the network with a batch size of {dataloader_train.batch_size} per GPU ...')
    logger.info(f'Progress will be displayed every {verbose_iters} batch iterations and the model will be saved every 10.0 minutes.')

    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    num_batches_per_epoch = len(dataloader_train)
    # Debugging statement before the loop
    logger.info('Starting training loop...')
    
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=fig_dir)

    for epoch in tqdm(range(cur_epoch, total_epochs), desc="Epochs"):
    # for epoch in tqdm(range(1), desc="Epochs"):
        
        epoch_loss_total = 0
        epoch_reconstruction_loss = 0
        epoch_consistency_loss = 0
        epoch_batches = 0
        
        batch_progress = tqdm(get_train_samples(dataloader_train, train_nested_batches), total=num_batches_per_epoch, desc="Batches")
        for batch_idx, (samples, masks, ra_decs) in enumerate(batch_progress):
            # if batch_idx >= 1:
            #     break
            samples = samples.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            ra_decs = ra_decs.to(device, non_blocking=True)
            
            logger.debug(f'Processing batch {batch_idx+1}/{num_batches_per_epoch}...')
            try:
                student_model, optimizer, lr_scheduler, losses_cp = run_iter(student_model, teacher_model, samples, ra_decs, masks,
                                                                             mask_ratio, optimizer,
                                                                             lr_scheduler,
                                                                             losses_cp, mode='train')
            except Exception as e:
                logger.error(f'Error in run_iter for batch {batch_idx+1}: {e}')
                raise

            epoch_loss_total += losses_cp['train_total_loss'][-1]
            epoch_reconstruction_loss += losses_cp['train_reconstruction_loss'][-1]
            epoch_consistency_loss += losses_cp['train_consistency_loss'][-1]
            epoch_batches += 1
            
            
            




            if (epoch * num_batches_per_epoch + batch_idx) % verbose_iters == 0:
                with torch.no_grad():
                    val_loss_total = 0
                    val_loss_recon = 0
                    val_loss_cons = 0
                    val_batches = 0
                    for i, (samples, masks, ra_decs) in enumerate(dataloader_val):
                        if i >= 200:
                            break
                        samples = samples.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        ra_decs = ra_decs.to(device, non_blocking=True)

                        try:
                            student_model, optimizer, lr_scheduler, losses_cp = run_iter(student_model, teacher_model, samples, ra_decs, masks,
                                                                                         mask_ratio, optimizer,
                                                                                         lr_scheduler,
                                                                                         losses_cp, mode='val')
                        except Exception as e:
                            logger.error(f'Error in validation run_iter for batch {i+1}: {e}')
                            raise

                        val_loss_total += losses_cp['val_total_loss'][-1]
                        val_loss_recon += losses_cp['val_reconstruction_loss'][-1]
                        val_loss_cons += losses_cp['val_consistency_loss'][-1]
                        val_batches += 1
                        if i >= 1:
                            break

                    val_loss_total /= val_batches
                    val_loss_recon /= val_batches
                    val_loss_cons /= val_batches

                    if lp_class_data_file or lp_regress_data_file:
                        linear_probe(student_model, losses_cp, device, dataloader_val,
                                     lp_class_data_file, lp_regress_data_file, combine=lp_combine)

                # Calculate averages
                for k in losses_cp.keys():
                    if len(losses_cp[k]) > 0:
                        avg_loss = np.mean(np.array(losses_cp[k]), axis=0)
                        losses[k].append(avg_loss)
                    else:
                        losses[k].append(0.0)  # Add zero if no values are present
                losses['batch_iters'].append(epoch * num_batches_per_epoch + batch_idx)

                logger.info(f'\nEpoch: {epoch}, Batch: {batch_idx}/{num_batches_per_epoch}')
                logger.info('Losses:')
                logger.info(f'\tTraining Dataset\n\t\tTotal Loss: {losses["train_total_loss"][-1]:0.3f}')
                logger.info(f'\tValidation Dataset\n\t\tTotal Loss: {val_loss_total:0.3f}')
                
                if lp_class_data_file or lp_regress_data_file:
                    logger.info('Linear Probing Results:')
                    if lp_class_data_file:
                        logger.info(f'\tClassification Accuracy:\n\t\tTraining: {losses["train_lp_acc"][-1]:0.3f}, Validation: {losses["val_lp_acc"][-1]:0.3f}')
                    if lp_regress_data_file:
                        logger.info(f'\tRegression R2\n\t\tTraining: {losses["train_lp_r2"][-1]:0.3f}, Validation: {losses["val_lp_r2"][-1]:0.3f}')
                
                losses_cp = defaultdict(list)

                # Debugging statements
                print(f"Batch iters length: {len(losses['batch_iters'])}")
                print(f"Train total loss length: {len(losses['train_total_loss'])}")
                if 'val_total_loss' in losses.keys():
                    print(f"Val total loss length: {len(losses['val_total_loss'])}")

                if len(losses['batch_iters']) > 1:
                    # Plot progress
                    plot_progress(losses, y_lims=[(0, 1), (0, 1), (0, 1), (0, 1)],
                                  savename=os.path.join(fig_dir,
                                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
                else:
                    print("Not enough data to plot progress")
                # Plot 5 validation samples
                pred_imgs, mask_imgs, orig_imgs = mae_predict(student_model, dataloader_val,
                                                              device,
                                                              mask_ratio,
                                                              single_batch=True)
                plot_batch(orig_imgs, mask_imgs, pred_imgs, n_samples=5, channel_index=0,
                           savename=os.path.join(fig_dir,
                                                 f'{os.path.basename(model_filename).split(".")[0]}_{epoch * num_batches_per_epoch + batch_idx}iters.png'))

        average_epoch_loss_total = epoch_loss_total / epoch_batches
        average_epoch_reconstruction_loss = epoch_reconstruction_loss / epoch_batches
        average_epoch_consistency_loss = epoch_consistency_loss / epoch_batches

        # Log the average losses
        logger.info(f'Epoch: {epoch}, Average Total Loss: {average_epoch_loss_total:.4f}')
        logger.info(f'Average Reconstruction Loss: {average_epoch_reconstruction_loss:.4f}')
        logger.info(f'Average Consistency Loss: {average_epoch_consistency_loss:.4f}')

        # Validation loss calculation
        with torch.no_grad():
            val_loss_total = 0
            val_loss_recon = 0
            val_loss_cons = 0
            val_batches = 0
            for i, (samples, masks, ra_decs) in enumerate(dataloader_val):
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                ra_decs = ra_decs.to(device, non_blocking=True)

                try:
                    student_model, optimizer, lr_scheduler, losses_cp = run_iter(student_model, teacher_model, samples, ra_decs, masks,
                                                                                 mask_ratio, optimizer,
                                                                                 lr_scheduler,
                                                                                 losses_cp, mode='val')
                except Exception as e:
                    logger.error(f'Error in validation run_iter for batch {i+1}: {e}')
                    raise

                val_loss_total += losses_cp['val_total_loss'][-1]
                val_loss_recon += losses_cp['val_reconstruction_loss'][-1]
                val_loss_cons += losses_cp['val_consistency_loss'][-1]
                val_batches += 1
                if i >= 200:
                    break

            val_loss_total /= val_batches
            val_loss_recon /= val_batches
            val_loss_cons /= val_batches

        logger.info(f'Epoch {epoch} completed. Average Training Total Loss: {epoch_loss_total:.6f}, Average Training Reconstruction Loss: {epoch_reconstruction_loss:.6f}, Average Training Consistency Loss: {epoch_consistency_loss:.6f}')
        logger.info(f'Average Validation Total Loss: {val_loss_total:.6f}, Average Validation Reconstruction Loss: {val_loss_recon:.6f}, Average Validation Consistency Loss: {val_loss_cons:.6f}')
        
        # Plot progress
        if len(losses['batch_iters']) > 1:
            plot_progress(losses, y_lims=[(0, 1), (0, 1), (0, 1), (0, 1)],
                          savename=os.path.join(fig_dir,
                                                f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))

        # Save model after each epoch
        logger.info('Saving network...')
        try:
            torch.save({'epoch': epoch,
                        'batch_iters': (epoch + 1) * num_batches_per_epoch,
                        'losses': losses,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'model': student_model.module.state_dict()},
                       model_filename)
        except Exception as e:
            logger.error(f'Error saving network: {e}')
            raise
        

    logger.info('Training complete.')
    
# Run the training
if __name__ == "__main__":
    # args = parseArguments()
    # args = args.parse_args()
    args = {"model_name":'mim_MOCO_32', "verbose_iters":50, "cp_time":10.0, "data_dir":None}
    class khers:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    def __str__(self):
        return str(self.__dict__)
    args = khers(**args)
    print(args)
    main(args)
    
    # Close the TensorBoard writer
    writer.close()
    # import trace; import sys;
    # #     # Set up the tracer
    # # tracer = trace.Trace(
    # #     ignoredirs=[sys.prefix, sys.exec_prefix],
    # #     trace=True,
    # #     count=False,
    # #     outfile='trace_output.txt'
    # # )

    # # # Run the main function with the tracer
    # # tracer.run('main(args)')
    
    
    #     # Initialize tracer
    # tracer = trace.Trace(
    #     ignoredirs=[sys.prefix, sys.exec_prefix],
    #     trace=True,
    #     count=False,
    #     outfile='trace_output.txt'  # Specify the output file
    # )
    
    # try:
    #     # Run the main function with the tracer
    #     tracer.run('main(args)')
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     raise

    # # logger.info('\nTraining complete.')
