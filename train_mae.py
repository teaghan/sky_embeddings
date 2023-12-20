import os
import numpy as np
import time
import configparser
from collections import defaultdict
import torch

from utils.pretrain import run_iter, parseArguments
from utils.models_mae import build_model
from utils.dataloader import build_dataloader

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Using Torch version: %s' % (torch.__version__))
    print('Using a %s device' % (device))

    # Directories
    cur_dir = os.path.dirname(__file__)
    config_dir = os.path.join(cur_dir, 'configs/')
    model_dir = os.path.join(cur_dir, 'models/')
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(cur_dir, 'data/')

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
    dataloader_train = build_dataloader(os.path.join(data_dir, config['DATA']['train_data_file']), 
                                        config['DATA']['norm_type'], 
                                        int(config['TRAINING']['batch_size']), 
                                        int(config['TRAINING']['num_workers']), 
                                        shuffle=True)


    print('The training set consists of %i cutouts.' % (len(dataloader_train.dataset)))
    
    train_network(model, dataloader_train, optimizer, 
                  lr_scheduler, device,
                  float(config['TRAINING']['mask_ratio']),
                  losses, cur_iter, 
                  int(float(config['TRAINING']['total_batch_iters'])),
                  args.verbose_iters, args.cp_time)

def train_network(model, dataloader_train, optimizer, lr_scheduler, device, mask_ratio, losses, 
                  cur_iter, total_batch_iters, verbose_iters, cp_time):
    print('Training the network with a batch size of %i ...' % (dataloader_train.batch_size))
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

                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(np.array(losses_cp[k]), axis=0))
                losses['batch_iters'].append(cur_iter)
                
                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Losses:')
                print('\tTraining Dataset')
                print('\t\tTotal Loss: %0.3f'% (losses['train_loss'][-1]))

                # Reset checkpoint loss dictionary
                losses_cp = defaultdict(list)

            # Increase the iteration
            cur_iter += 1

            if (time.time() - cp_start_time) >= cp_time*60:
                
                # Save periodically
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.state_dict()},
                                model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'lr_scheduler' : lr_scheduler.state_dict(),
                                'model' : model.state_dict()},
                                model_filename)
                # Finish training
                break 
                
# Run the training
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTraining complete.')