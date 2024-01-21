import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch

from utils.pretrain import run_iter, parseArguments
from utils.models_mae import build_model
from utils.dataloader import build_dataloader
from utils.analysis_fns import plot_progress, mae_predict, plot_batch

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
    num_workers = min([os.cpu_count(),12*max([n_gpu,1])])
    dataloader_train = build_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                        norm_type=config['DATA']['norm_type'], 
                                        batch_size=int(int(config['TRAINING']['batch_size'])/n_gpu), 
                                        num_workers=num_workers,
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        shuffle=True)

    dataloader_val = build_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                      norm_type=config['DATA']['norm_type'], 
                                      batch_size=int(int(config['TRAINING']['batch_size'])/n_gpu), 
                                      num_workers=num_workers,
                                      img_size=int(config['ARCHITECTURE']['img_size']),
                                      shuffle=True)


    print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
    
    train_network(model, dataloader_train, dataloader_val, 
                  optimizer, lr_scheduler, device,
                  float(config['TRAINING']['mask_ratio']),
                  losses, cur_iter, 
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time, model_filename, fig_dir)

def train_network(model, dataloader_train, dataloader_val, optimizer, lr_scheduler, device, mask_ratio, 
                  losses, cur_iter, total_batch_iters, verbose_iters, cp_time, model_filename, fig_dir):
    print('Training the network with a batch size of %i per GPU ...' % (dataloader_train.batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while cur_iter < (total_batch_iters):
        # Iterate through training dataset
        for train_samples, _ in dataloader_train:
            
            # Switch to GPU if available
            train_samples = train_samples.to(device, non_blocking=True)
            
            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, train_samples, 
                                                                 mask_ratio, optimizer, 
                                                                 lr_scheduler, 
                                                                 losses_cp, mode='train')
            
                            
            # Evaluate validation set and display losses
            if cur_iter % verbose_iters == 0:

                with torch.no_grad():
                    for i, (val_samples, _) in enumerate(dataloader_val):
                        # Switch to GPU if available
                        val_samples = val_samples.to(device, non_blocking=True)

                        # Run an iteration
                        model, optimizer, lr_scheduler, losses_cp = run_iter(model, val_samples, 
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
