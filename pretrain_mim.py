import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from utils.misc import str2bool, parseArguments
from utils.pretrain_fns import run_iter, linear_probe
from utils.mim_vit import build_model
from utils.dataloaders import build_unions_dataloader, build_h5_dataloader, build_fits_dataloader
from utils.plotting_fns import plot_progress, plot_batch, plot_batch_raw
from utils.eval_fns import mae_predict

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

    # Data loader stuff
    num_workers = min([os.cpu_count(),12*n_gpu])
    if num_workers>1:
        num_workers -=1
        
    # Masking stuff
    if 'mim' in config['ARCHITECTURE']['model_type']:
        mask_ratio = None
        max_mask_ratio = float(config['TRAINING']['max_mask_ratio'])
    else:
        mask_ratio = float(config['TRAINING']['mask_ratio'])
        max_mask_ratio = None

    # Linear probing validation data files
    lp_class_data_file = os.path.join(data_dir, config['DATA']['lp_class_data_file']) if 'lp_class_data_file' in config['DATA'] else None
    lp_regress_data_file = os.path.join(data_dir, config['DATA']['lp_regress_data_file']) if 'lp_regress_data_file' in config['DATA'] else None

    # Build dataloaders
    if 'survey' in config['DATA'] and config['DATA']['survey'] == 'UNIONS': 
        # using Nick's data streaming method for pre-training dataloader
        dataloader_train = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=max_mask_ratio, 
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches)
        print('The training set streaming has begun') 
        train_nested_batches = False

        # all other dataloaders are from already saved HDF5 files
        dataloader_val = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=max_mask_ratio, eval=True,
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                eval_data_file=(config['DATA']['val_data_file']))   

        if not lp_regress_data_file == None:   
            dataloader_regress = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=0.0, eval=True,
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                label_keys=['zspec'],
                                                eval_data_file=(config['DATA']['lp_regress_data_file']))
        else: 
            dataloader_regress = None
            
        if not lp_class_data_file == None:
            dataloader_classfication = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=0.0, eval=True,
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                label_keys=['is_dwarf'],
                                                eval_data_file=(config['DATA']['lp_class_data_file']))
            
        else:
            dataloader_classfication = None
    
        print('dataloader_classfication:', dataloader_classfication)

    else: 
        if 'train_data_file' in config['DATA']:
            # Using .h5 training file
            dataloader_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                                    batch_size=int(config['TRAINING']['batch_size']), 
                                                    num_workers=num_workers,
                                                    patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                                    num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                                    max_mask_ratio=max_mask_ratio, 
                                                    img_size=int(config['ARCHITECTURE']['img_size']),
                                                    num_patches=model.module.patch_embed.num_patches,
                                                    shuffle=True)
            print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
            train_nested_batches = False
        else:
            # Using fits files in training directory
            # Might need to decrease num_workers and increase cutouts_per_tile
            dataloader_train =  build_fits_dataloader(eval(config['DATA']['train_data_paths']), 
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
                                            num_patches=model.module.patch_embed.num_patches,
                                            shuffle=True)

        # just a band-aid fix for now
        dataloader_regress, dataloader_classfication = dataloader_val, dataloader_val

    
    train_network(model, dataloader_train, dataloader_val, dataloader_regress, dataloader_classfication,
                  train_nested_batches, optimizer, lr_scheduler, device,
                  mask_ratio,
                  losses, cur_iter, 
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time, model_filename, fig_dir,
                  lp_class_data_file, lp_regress_data_file, config['DATA']['lp_combine'])

def get_train_samples(dataloader, train_nested_batches):
    '''Accomodates both dataloaders.'''
    if train_nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks, ra_decs in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask, ra_dec in zip(sample_batches[0], masks[0], ra_decs[0]):
                yield samples, mask, ra_dec
    else:
        for samples, mask, ra_dec in dataloader:
            yield samples, mask, ra_dec

def train_network(model, dataloader_train, dataloader_val, dataloader_regress, dataloader_classification, 
                  train_nested_batches, optimizer, lr_scheduler, device, mask_ratio, 
                  losses, cur_iter, total_batch_iters, verbose_iters, cp_time, model_filename, fig_dir, 
                  lp_class_data_file, lp_regress_data_file, lp_combine):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    print('lp_class_data_file:', lp_class_data_file)
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    #time1 = time.time()
    while cur_iter < (total_batch_iters):
        print('cur_iter:', cur_iter)

        # Iterate through training dataset
        # (this might return a batch and then it just iterates over that?)
        for samples, masks, ra_decs in get_train_samples(dataloader_train, train_nested_batches):
            samples = samples.squeeze(1)
            masks = masks.squeeze(1)
            ra_decs = ra_decs.squeeze(1)
            print('batch loaded.')
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            ra_decs = ra_decs.to(device, non_blocking=True)
            print('batch on gpu.')
            
            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, ra_decs, masks,
                                                                 mask_ratio, optimizer, 
                                                                 lr_scheduler, 
                                                                 losses_cp, mode='train')
            print('training iter done.')
            
            #if cur_iter % 100 == 0:
            #    time_el = time.time()-time1
            #    print(f'{time_el:0.1f} seconds elapsed.')
            #    time1 = time.time()
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:
                print(cur_iter, verbose_iters)

                with torch.no_grad():
                    # Calculate average loss on validation set
                    for i, (samples, masks, ra_decs) in enumerate(dataloader_val):
                        # Switch to GPU if available
                        samples = samples.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        ra_decs = ra_decs.to(device, non_blocking=True)

                        # Run an iteration
                        model, optimizer, lr_scheduler, losses_cp = run_iter(model, samples, ra_decs, masks,
                                                                             mask_ratio, optimizer, 
                                                                             lr_scheduler, 
                                                                             losses_cp, mode='val')
                        # Don't bother with the whole dataset
                        if i>=1000:
                            break
                
                    if lp_class_data_file or lp_regress_data_file:
                        # Run Linear Probing tests
                        linear_probe(model, losses_cp, device, dataloader_regress, dataloader_classification,
                                     lp_class_data_file, lp_regress_data_file, combine=lp_combine)
                        print(lp_class_data_file)
                
                # Calculate averages
                for k in losses_cp.keys():
                    print(k)
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                print('\tValidation Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                if lp_class_data_file or lp_regress_data_file:
                    print('Linear Probing Results:')
                    if lp_class_data_file:
                        print('\tClassification Accuracy:') # these keys are missing
                        print('\t\tTraining: %0.3f, Validation: %0.3f'% (losses['train_lp_acc'][-1], losses['val_lp_acc'][-1]))
                    if lp_regress_data_file:
                        print('\tRegression R2')
                        print('\t\tTraining: %0.3f, Validation: %0.3f'% (losses['train_lp_r2'][-1], losses['val_lp_r2'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)
                
                if len(losses['batch_iters'])>1:
                    # Plot progress
                    plot_progress(losses, y_lims=[(0,0.7), (0.8,1.), (0.6,1.)], 
                                  savename=os.path.join(fig_dir, 
                                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
                # Plot 5 validation samples
                pred_imgs, mask_imgs, orig_imgs = mae_predict(model, dataloader_val, 
                                                              device, 
                                                              mask_ratio, 
                                                              single_batch=True)
                
                # make these more than just same old star-like thing (picked for UNIONS)
                # [specific selection here]
                interesting_val_idx = [0, 1, 6, 7, 8, 10, 12]
                plot_batch(orig_imgs[interesting_val_idx], mask_imgs[interesting_val_idx], pred_imgs[interesting_val_idx], n_samples=len(interesting_val_idx), 
                                                channel_index=0, savename=os.path.join(fig_dir, 
                                                 f'{os.path.basename(model_filename).split(".")[0]}_{cur_iter}iters_val.png'))
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
