import os
# Directory of training script
cur_dir = os.path.dirname(__file__)

import numpy as np
import time
import configparser
from collections import defaultdict

import torch
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.pretrain import run_iter, parseArguments, str2bool
import utils.models_mae
from utils.models_mae import load_model
from utils.dataloader import CutoutDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_gpus = torch.cuda.device_count()

torch.backends.cudnn.benchmark = True

print('Using Torch version: %s' % (torch.__version__))
print('Using a %s device with %i gpus' % (device, num_gpus))

# Collect the command line arguments
args = parseArguments()
args = args.parse_args()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir

# Directories
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')

# Training parameters from config file
train_data_file = os.path.join(data_dir, config['DATA']['train_data_file'])
channel_means = eval(config['DATA']['channel_means'])
channel_stds = eval(config['DATA']['channel_stds'])
pretrained_start = str2bool(config['TRAINING']['pretrained_start'])
batch_size = int(config['TRAINING']['batch_size'])
total_batch_iters = int(config['TRAINING']['total_batch_iters'])
mask_ratio = float(config['TRAINING']['mask_ratio'])
norm_pix_loss = str2bool(config['TRAINING']['norm_pix_loss'])
weight_decay = float(config['TRAINING']['weight_decay'])
init_lr = float(config['TRAINING']['init_lr'])
final_lr_factor = float(config['TRAINING']['final_lr_factor'])
num_workers = int(config['TRAINING']['num_workers'])

# Model architecture
img_size = int(config['ARCHITECTURE']['img_size'])
model_type = config['ARCHITECTURE']['model_type']

# Construct the model
model = utils.models_mae.__dict__[model_type](img_size=img_size,
                                              norm_pix_loss=norm_pix_loss)
model.to(device)

# Set weight decay to 0 for bias and norm layers
param_groups = optim_factory.param_groups_weight_decay(model, weight_decay)

# Optimizer
optimizer = torch.optim.AdamW(param_groups, lr=init_lr, betas=(0.9, 0.95))

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr,
                                                   total_steps=int(total_batch_iters), 
                                                   pct_start=0.05, anneal_strategy='cos', 
                                                   cycle_momentum=True, 
                                                   base_momentum=0.85, 
                                                   max_momentum=0.95, div_factor=25.0, 
                                                   final_div_factor=final_lr_factor, 
                                                   three_phase=False)

loss_scaler = NativeScaler()

if pretrained_start:
    # Load pretrained ViT model weights here...
    pass

# Load model state from previous training (if any)
model_filename =  os.path.join(model_dir, model_name+'.pth.tar')    
model, losses, cur_iter = load_model(model, model_filename, optimizer, loss_scaler)


# Data loaders
transform_train = transforms.Compose([
            #transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=channel_means, std=channel_stds)])

dataset_train = CutoutDataset(train_data_file, transform=transform_train)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                      batch_size=batch_size, 
                                                      shuffle=True, 
                                                      num_workers=num_workers,
                                                      pin_memory=True)

print('The training set consists of %i cutouts.' % (len(dataset_train)))

def train_network(model, optimizer, lr_scheduler, loss_scaler, cur_iter):
    print('Training the network with a batch size of %i ...' % (batch_size))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    while cur_iter < (total_batch_iters):
        # Iterate through both training datasets simultaneously
        for train_samples, _ in dataloader_train:
            
            # Switch to GPU if available
            train_samples = train_samples.to(device, non_blocking=True)
            
            # Run an iteration of training
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, train_samples, 
                                                                 mask_ratio, optimizer, 
                                                                 lr_scheduler, loss_scaler, 
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
                                'loss_scaler': loss_scaler.state_dict(),
                                'model' : model.state_dict()},
                                model_filename)

                cp_start_time = time.time()

            if cur_iter > total_batch_iters:
                
                # Save after training
                print('Saving network...')
                torch.save({'batch_iters': cur_iter,
                                'losses': losses,
                                'optimizer' : optimizer.state_dict(),
                                'loss_scaler': loss_scaler.state_dict(),
                                'model' : model.state_dict()},
                                model_filename)
                # Finish training
                break 
                
# Run the training
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, loss_scaler, cur_iter)
    print('\nTraining complete.')