import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch

from utils.misc import str2bool, parseArguments, select_training_indices
from utils.predictor_training_fns import run_iter
from utils.vit import build_model
from utils.dataloaders import build_h5_dataloader
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
    mae_name = config['TRAINING']['pretained_mae']
    if mae_name=='None':
        mae_filename = 'None'
        mae_config = config
    else:
        # Load pretrained MAE configuration
        mae_config = configparser.ConfigParser()
        mae_config.read(config_dir+mae_name+'.ini')
        mae_filename =  os.path.join(model_dir, mae_name+'.pth.tar')
    if os.path.exists(model_filename.replace('.pth.tar', '_best.pth.tar')):
        model, losses, cur_iter, optimizer, lr_scheduler = build_model(config, mae_config, 
                                                                   model_filename.replace('.pth.tar', '_best.pth.tar'), mae_filename,
                                                                   device, build_optimizer=True)
    else:
        model, losses, cur_iter, optimizer, lr_scheduler = build_model(config, mae_config, 
                                                                   model_filename, mae_filename,
                                                                   device, build_optimizer=True)
    
    # Data loaders    
    num_workers = min([os.cpu_count(),12*n_gpu])
    if num_workers>1:
        num_workers -=1

    num_train = int(config['TRAINING']['num_train'])
    if num_train>-1:
        if 'crossentropy' in config['TRAINING']['loss_fn'].lower():
            train_indices = select_training_indices(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                                    num_train, balanced=False)
        else:    
            train_indices = range(num_train)
    else:
        train_indices = None

    dataloader_train = build_h5_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                           batch_size=int(config['TRAINING']['batch_size']), 
                                           num_workers=num_workers,
                                           label_keys=eval(config['DATA']['label_keys']),
                                           img_size=int(config['ARCHITECTURE']['img_size']),
                                           patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                           num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                           num_patches=model.module.patch_embed.num_patches,
                                           augment=str2bool(config['TRAINING']['augment']),
                                           brightness=float(config['TRAINING']['brightness']), 
                                           noise=float(config['TRAINING']['noise']), 
                                           nan_channels=int(config['TRAINING']['nan_channels']),
                                           shuffle=True,
                                           indices=train_indices)
    
    dataloader_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                        batch_size=int(config['TRAINING']['batch_size']), 
                                        num_workers=num_workers,
                                        label_keys=eval(config['DATA']['label_keys']),
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                         num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                        num_patches=model.module.patch_embed.num_patches,
                                        shuffle=True)
    
    print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))

    train_network(model, dataloader_train, dataloader_val, 
                  optimizer, lr_scheduler, device,
                  losses, cur_iter, 
                  config['TRAINING']['loss_fn'],
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time, model_filename, fig_dir,
                 str2bool(config['TRAINING']['use_label_errs']))

def train_network(model, dataloader_train, dataloader_val, optimizer, lr_scheduler, device, 
                  losses, cur_iter, loss_fn, total_batch_iters, verbose_iters, cp_time, model_filename, fig_dir,
                  use_label_errs):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))

    if 'val_loss' in losses.keys():
        best_val_loss = np.min(losses['val_loss'])
    else:
        best_val_loss = np.inf
    did_not_improve_count = 0
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while (cur_iter < (total_batch_iters)) & (did_not_improve_count<50):
        # Iterate through training dataset
        for input_samples, sample_masks, ra_decs, sample_labels in dataloader_train:
            
            # Switch to GPU if available
            input_samples = input_samples.to(device, non_blocking=True)
            sample_masks = sample_masks.to(device, non_blocking=True)
            ra_decs = ra_decs.to(device, non_blocking=True)
            sample_labels = sample_labels.to(device, non_blocking=True)

            if use_label_errs:
                # Collect label uncertainties
                num_labels = sample_labels.size(1)//2
                sample_label_errs = sample_labels[:,num_labels:]
                sample_labels = sample_labels[:,:num_labels]
            else:
                sample_label_errs = None
            
            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, input_samples, sample_masks, ra_decs, sample_labels,
                                                                 optimizer, 
                                                                 lr_scheduler, 
                                                                 losses_cp, 
                                                                 loss_fn,
                                                                 label_uncertainties=sample_label_errs,
                                                                 mode='train')
                            
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:

                with torch.no_grad():
                    for i, (input_samples, sample_masks, ra_decs, sample_labels) in enumerate(dataloader_val):
                        # Switch to GPU if available
                        input_samples = input_samples.to(device, non_blocking=True)
                        sample_masks = sample_masks.to(device, non_blocking=True)
                        ra_decs = ra_decs.to(device, non_blocking=True)
                        sample_labels = sample_labels.to(device, non_blocking=True)


                        if use_label_errs:
                            # Collect label uncertainties
                            num_labels = sample_labels.size(1)//2
                            sample_label_errs = sample_labels[:,num_labels:]
                            sample_labels = sample_labels[:,:num_labels]
                        else:
                            sample_label_errs = None

                        # Run an iteration
                        model, optimizer, lr_scheduler, losses_cp = run_iter(model, input_samples, 
                                                                             sample_masks, ra_decs, sample_labels,
                                                                             optimizer, 
                                                                             lr_scheduler, 
                                                                             losses_cp, 
                                                                             loss_fn,
                                                                             label_uncertainties=sample_label_errs,
                                                                             mode='val')
                        # Don't bother with the whole dataset
                        #if i>=200:
                        #    break
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))
                if 'mse' in loss_fn.lower():
                    print('\t\tMAE: %0.3f'% (losses['train_mae'][-1]))
                else:
                    print('\t\tAccuracy: %0.3f'% (losses['train_acc'][-1]))
                print('\tValidation Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['val_loss'][-1]))
                if 'mse' in loss_fn.lower():
                    print('\t\tMAE: %0.3f'% (losses['val_mae'][-1]))
                else:
                    print('\t\tAccuracy: %0.3f'% (losses['val_acc'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)
                
                if len(losses['batch_iters'])>1:

                    if 'mse' in loss_fn.lower():
                        y_lims = [(0,0.005), (0,0.1)]
                    else:
                        y_lims = [(0,0.2), (0.7,1)]
                    # Plot progress
                    plot_progress(losses, y_lims=y_lims, 
                                  savename=os.path.join(fig_dir, 
                                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))

                # Save best model
                if losses['val_loss'][-1]<best_val_loss:
                    print('\t%0.3f, %0.3f'% (best_val_loss, losses['val_loss'][-1]))
                    best_val_loss = losses['val_loss'][-1]
                    print('Saving network...')
                    torch.save({'batch_iters': cur_iter,
                                    'losses': losses,
                                    'optimizer' : optimizer.state_dict(),
                                    'lr_scheduler' : lr_scheduler.state_dict(),
                                    'model' : model.module.state_dict()},
                                    model_filename.replace('.pth.tar', '_best.pth.tar'))
                    did_not_improve_count = 0
                else:
                    did_not_improve_count += 1

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
