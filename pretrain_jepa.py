import os
import numpy as np
import copy
import time
import configparser
from collections import defaultdict
import torch
from torch.nn import DataParallel
#from torch.nn.parallel import DistributedDataParallel
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from utils.misc import str2bool, parseArguments
from utils.pretrain_engine import run_iter, linear_probe
from utils.vision_transformer import build_model
from utils.masks import SimpleMaskCollator, MaskCollator
from utils.data import build_h5_dataloader, build_fits_dataloader, get_augmentations
from utils.plotting_fns import plot_progress

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_gpu = torch.cuda.device_count()

    print(f'Using Torch version: {torch.__version__}')
    print(f'Using a {device} device with {n_gpu} GPU(s)')


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
    config.read(config_dir+model_name+'.ini')

    # Display model configuration
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))

    # Construct the model, optimizer, etc.
    model_filename =  os.path.join(model_dir, model_name+'.pth.tar') 
    (encoder, predictor, losses, cur_iter, 
     optimizer, lr_scheduler, wd_scheduler, momentum_scheduler) = build_model(config, model_filename, 
                                                                   device, build_optimizer=True)
    target_encoder = copy.deepcopy(encoder)

    # MultiGPU
    encoder = DataParallel(encoder)
    predictor = DataParallel(predictor)
    target_encoder = DataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    
    # Masking and data loading
    mask_collator = SimpleMaskCollator(
        input_size=int(config['ARCHITECTURE']['img_size']),
        patch_size=int(config['ARCHITECTURE']['patch_size']),
        nenc=int(config['MASK']['num_enc_masks']),
        npred=int(config['MASK']['num_pred_masks']))
    '''mask_collator = MaskCollator(
        input_size=int(config['ARCHITECTURE']['img_size']),
        patch_size=int(config['ARCHITECTURE']['patch_size']),
        pred_mask_scale=eval(config['MASK']['pred_mask_scale']),
        enc_mask_scale=eval(config['MASK']['enc_mask_scale']),
        aspect_ratio=eval(config['MASK']['aspect_ratio']),
        nenc=int(config['MASK']['num_enc_masks']),
        npred=int(config['MASK']['num_pred_masks']),
        allow_overlap=str2bool(config['MASK']['allow_overlap']),
        min_keep=int(config['MASK']['min_keep']))'''
    
    transform = None#get_augmentations(img_size=int(config['ARCHITECTURE']['img_size']), 
                #                  flip=False, crop=False, brightness=True, noise=True, nan_channels=True)
    
    num_workers = min([os.cpu_count(),12*n_gpu])
    if num_workers>1:
        num_workers -=1
    if n_gpu>1:
        batch_size = int(int(config['TRAINING']['batch_size'])/n_gpu)
    else:
        batch_size = int(config['TRAINING']['batch_size'])

    # Build dataloaders
    if 'train_data_file' in config['DATA']:
        # Using .h5 training file
        dataloader_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                                batch_size=batch_size, 
                                                num_workers=num_workers,
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                pos_channel=str2bool(config['DATA']['pos_channel']), 
                                                transforms=transform,
                                               collator=mask_collator,
                                               shuffle=True)
        train_nested_batches = False
        print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
    else:
        # Using fits files in training directory
        # Might need to decrease num_workers and increase cutouts_per_tile
        dataloader_train =  build_fits_dataloader(eval(config['DATA']['train_data_paths']), 
                                                  bands=eval(config['DATA']['bands']), 
                                                  min_bands=int(config['DATA']['min_bands']), 
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  img_size=int(config['ARCHITECTURE']['img_size']), 
                                                  cutouts_per_tile=int(config['DATA']['cutouts_per_tile']), 
                                                  use_calexp=str2bool(config['DATA']['use_calexp']),
                                                  transforms=transform, 
                                                  collator=mask_collator,
                                                  shuffle=True)
        train_nested_batches = True
    
    dataloader_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                          batch_size=batch_size, 
                                          num_workers=num_workers,
                                          img_size=int(config['ARCHITECTURE']['img_size']),
                                          pos_channel=str2bool(config['DATA']['pos_channel']),
                                         collator=mask_collator,
                                          shuffle=True)

    # Linear probing validation data files
    lp_class_data_file = os.path.join(data_dir, config['DATA']['lp_class_data_file']) if 'lp_class_data_file' in config['DATA'] else None
    lp_regress_data_file = os.path.join(data_dir, config['DATA']['lp_regress_data_file']) if 'lp_regress_data_file' in config['DATA'] else None
        
    train_network(encoder, predictor, target_encoder, 
                  optimizer, lr_scheduler, wd_scheduler, momentum_scheduler,
                  dataloader_train, dataloader_val, train_nested_batches,
                  device, losses, cur_iter, 
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time, model_filename, fig_dir,
                  lp_class_data_file, lp_regress_data_file)

def get_train_samples(dataloader, train_nested_batches):
    '''Accomodates both dataloaders.'''
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks_enc_batches, masks_pred_batches in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, masks_enc, masks_pred in zip(sample_batches, masks_enc_batches, masks_pred_batches):
                yield samples, masks_enc, masks_pred
    else:
        for samples, masks_enc, masks_pred in dataloader:
            yield samples, masks_enc, masks_pred

def train_network(encoder, predictor, target_encoder, 
                  optimizer, lr_scheduler, wd_scheduler, momentum_scheduler,
                  dataloader_train, dataloader_val, train_nested_batches,
                  device, losses, cur_iter, total_batch_iters, 
                  verbose_iters, cp_time, model_filename, fig_dir, 
                  lp_class_data_file, lp_regress_data_file):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    #time1 = time.time()
    while cur_iter < (total_batch_iters):

        # Iterate through training dataset
        for imgs, masks_enc, masks_pred in get_train_samples(dataloader_train, train_nested_batches):

            # Switch to GPU if available
            imgs = imgs.to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            
            # Run an iteration of training
            (encoder, predictor, target_encoder, 
               optimizer, lr_scheduler, wd_scheduler, losses_cp) = run_iter(imgs, masks_enc, masks_pred, 
                                                                 encoder, predictor, target_encoder, 
                                                                 optimizer, lr_scheduler, wd_scheduler, 
                                                                 momentum_scheduler, 
                                                                 cur_iter, losses_cp, mode='train')
            
            #if cur_iter % 100 == 0:
            #    time_el = time.time()-time1
            #    print(f'{time_el:0.1f} seconds elapsed.')
            #    time1 = time.time()
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:

                with torch.no_grad():
                    # Calculate average loss on validation set
                    for i, (imgs, masks_enc, masks_pred) in enumerate(dataloader_val):
                         # Switch to GPU if available
                        imgs = imgs.to(device, non_blocking=True)
                        masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
                        masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

                        # Run an iteration
                        (encoder, predictor, target_encoder, 
               optimizer, lr_scheduler, wd_scheduler, losses_cp) = run_iter(imgs, masks_enc, masks_pred, 
                                                                             encoder, predictor, target_encoder, 
                                                                             optimizer, lr_scheduler, wd_scheduler, 
                                                                             momentum_scheduler, 
                                                                             cur_iter, losses_cp, mode='val')
                        # Don't bother with the whole dataset
                        if i>=200:
                            break
                
                    if lp_class_data_file or lp_regress_data_file:
                        # Run Linear Probing tests
                        linear_probe(encoder, losses_cp, device, dataloader_val, 
                                     lp_class_data_file, lp_regress_data_file)
                
                # Calculate averages
                for k in losses_cp.keys():
                    if k=='lr' or k=='wd':
                        losses[k].append(losses_cp[k][-1])
                    else:
                        losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('\nLearning Rate: %0.5f, Weight decay: %0.5f ' % (losses['lr'][-1], losses['wd'][-1]))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                print('\tValidation Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                if lp_class_data_file or lp_regress_data_file:
                    print('Linear Probing Results:')
                    if lp_class_data_file:
                        print('\tClassification Accuracy:')
                        print('\t\tTraining: %0.3f, Validation: %0.3f'% (losses['train_lp_acc'][-1], losses['val_lp_acc'][-1]))
                    if lp_regress_data_file:
                        print('\tRegression R2')
                        print('\t\tTraining: %0.3f, Validation: %0.3f'% (losses['train_lp_r2'][-1], losses['val_lp_r2'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)
                
                if len(losses['batch_iters'])>1:
                    # Plot progress
                    plot_progress(losses, y_lims=[(0,0.3), (0.5,1.), (0.,1.)], 
                                  savename=os.path.join(fig_dir, 
                                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
                    
            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time*60:
                
                # Save periodically
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'wd_scheduler' : wd_scheduler.state_dict(),
                                'encoder' : encoder.module.state_dict(),
                                'predictor' : predictor.module.state_dict()},
                                model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'wd_scheduler' : wd_scheduler.state_dict(),
                                'encoder' : encoder.module.state_dict(),
                                'predictor' : predictor.module.state_dict()},
                                model_filename)
                # Finish training
                break 

# Run the training
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)

print('\nTraining complete.')