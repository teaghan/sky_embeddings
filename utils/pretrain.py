import argparse
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    return parser


def run_iter(model, samples, mask_ratio, optimizer, lr_scheduler,
             losses_cp, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)
        
    # Calculate MAE loss
    #with torch.cuda.amp.autocast():
    loss, _, _ = model(samples, mask_ratio=mask_ratio)
        
    if 'train' in mode:
        if not torch.isnan(loss):
            # Update the gradients
            loss.backward()
            # Adjust network weights
            optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Adjust learning rate
        lr_scheduler.step()
        
        # Save loss and metrics
        losses_cp['train_loss'].append(float(loss))

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(loss))
                
    return model, optimizer, lr_scheduler, losses_cp