import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch

from utils.pretrain_simmim import str2bool, run_iter, parseArguments
from utils.models_simmim import build_model
from utils.dataloader_simmim import build_dataloader, build_fits_dataloader
from utils.analysis_fns_simmim import plot_progress, mae_predict, plot_batch

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
    model, losses, cur_iter, optimizer, lr_scheduler = build_model(config, model_filename, 
                                                                   device, build_optimizer=True)


    # Data loaders
    num_workers = min([os.cpu_count(),12*n_gpu])
    if num_workers>1:
        num_workers -=1
    if n_gpu>1:
        batch_size = int(int(config['TRAINING']['batch_size'])/n_gpu)
    else:
        batch_size = int(config['TRAINING']['batch_size'])
    if config['DATA']['norm_type']=='global':
        pix_mean = float(config['DATA']['pix_mean'])
        pix_std = float(config['DATA']['pix_std'])
    else:
        pix_mean = None
        pix_std = None

    if config['ARCHITECTURE']['model_type']=='simmim':
        mask_ratio = None
        max_mask_ratio = float(config['TRAINING']['max_mask_ratio'])
    else:
        mask_ratio = float(config['TRAINING']['mask_ratio'])
        max_mask_ratio = None
    if 'train_data_file' in config['DATA']:
        # Using .h5 training file
        dataloader_train = build_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                            norm_type=config['DATA']['norm_type'], 
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                            num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                            max_mask_ratio=max_mask_ratio, 
                                            img_size=int(config['ARCHITECTURE']['img_size']),
                                            pos_channel=str2bool(config['DATA']['pos_channel']), 
                                            pix_mean=pix_mean,
                                            pix_std=pix_std,
                                            num_patches=model.module.patch_embed.num_patches,
                                            shuffle=True)
        print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
        train_nested_batches = False
    else:
        # Using fits files in training directory
        # Might need to decrease num_workers and increase cutouts_per_tile
        dataloader_train =  build_fits_dataloader(eval(config['DATA']['train_data_paths']), 
                                                  bands=eval(config['DATA']['bands']), 
                                                  norm_type=config['DATA']['norm_type'], 
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                  max_mask_ratio=max_mask_ratio, 
                                                  img_size=int(config['ARCHITECTURE']['img_size']), 
                                                  cutouts_per_tile=int(config['DATA']['cutouts_per_tile']), 
                                                  pix_mean=pix_mean, 
                                                  pix_std=pix_std, 
                                                  augment=False, 
                                                  shuffle=True)
        train_nested_batches = True
    
    dataloader_val = build_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                        norm_type=config['DATA']['norm_type'], 
                                        batch_size=batch_size, 
                                        num_workers=num_workers,
                                      patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                      num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                      max_mask_ratio=max_mask_ratio, 
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        pos_channel=str2bool(config['DATA']['pos_channel']), 
                                        pix_mean=pix_mean,
                                        pix_std=pix_std, 
                                        num_patches=model.module.patch_embed.num_patches,
                                        shuffle=True)
    train_network(model, dataloader_train, dataloader_val, train_nested_batches,
                  optimizer, lr_scheduler, device,
                  mask_ratio,
                  losses, cur_iter, 
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time, model_filename, fig_dir)

def get_train_samples(dataloader, train_nested_batches):
    '''Accomodates both dataloaders.'''
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask in zip(sample_batches[0], masks[0]):
                yield samples, mask
    else:
        for samples, mask, _ in dataloader:
            yield samples, mask

def train_network(model, dataloader_train, dataloader_val, train_nested_batches, optimizer, lr_scheduler, device, mask_ratio, 
                  losses, cur_iter, total_batch_iters, verbose_iters, cp_time, model_filename, fig_dir):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while cur_iter < (total_batch_iters):

        # Iterate through training dataset
        for samples, masks in get_train_samples(dataloader_train, train_nested_batches):
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, masks,
                                                                 mask_ratio, optimizer, 
                                                                 lr_scheduler, 
                                                                 losses_cp, mode='train')
            
                            
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:

                with torch.no_grad():
                    for i, (samples, masks, _) in enumerate(dataloader_val):
                        # Switch to GPU if available
                        samples = samples.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)

                        # Run an iteration
                        model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, masks,
                                                                             mask_ratio, optimizer, 
                                                                             lr_scheduler, 
                                                                             losses_cp, mode='val')
                        # Don't bother with the whole dataset
                        if i>=100:
                            break
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                print('\tValidation Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)
                
                if len(losses['batch_iters'])>1:
                    # Plot progress
                    plot_progress(losses, y_lims=[(0,1.1)], 
                                  savename=os.path.join(fig_dir, 
                                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
                # Plot 5 validation samples
                pred_imgs, mask_imgs, orig_imgs = mae_predict(model, dataloader_val, 
                                                              device, 
                                                              mask_ratio, 
                                                              single_batch=True)
                plot_batch(orig_imgs, mask_imgs, pred_imgs, n_samples=5, channel_index=0,
                           savename=os.path.join(fig_dir, 
                                                 f'{os.path.basename(model_filename).split(".")[0]}_{cur_iter}iters.png'))

            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time*60:
                
                # Save periodically
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.module.state_dict()},
                                model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.module.state_dict()},
                                model_filename)
                # Finish training
                break 

# Run the training
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTraining complete.')
