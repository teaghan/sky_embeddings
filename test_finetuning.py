import os
import numpy as np
import configparser
import torch

from utils.finetune import parseArguments
from utils.models_vit import build_model
from utils.dataloader import build_dataloader
from utils.analysis_fns import plot_progress, ft_predict, plot_resid_hexbin

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
    fig_dir = os.path.join(cur_dir, 'figures/')

    # Load model configuration
    model_name = args.model_name
    config = configparser.ConfigParser()
    config.read(config_dir+model_name+'.ini')

    # ... and pretrained MAE configuration
    mae_name = config['TRAINING']['pretained_mae']
    mae_config = configparser.ConfigParser()
    mae_config.read(config_dir+mae_name+'.ini')

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
    mae_filename =  os.path.join(model_dir, mae_name+'.pth.tar') 
    model, losses, cur_iter = build_model(config, mae_config, 
                                          model_filename, mae_filename,
                                          device, build_optimizer=False)
    
    # Data loaders
    dataloader_val = build_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                        norm_type=mae_config['DATA']['norm_type'], 
                                        batch_size=int(config['TRAINING']['batch_size']), 
                                        num_workers=int(config['TRAINING']['num_workers']),
                                        label_keys=eval(config['DATA']['label_keys']),
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        shuffle=True)
    
    print('The validation set consists of %i cutouts.' % (len(dataloader_val.dataset)))

    tgt_labels, pred_labels = ft_predict(model, dataloader_val, device)
    
    plot_resid_hexbin([r'$z$'], tgt_labels, pred_labels,
                      x_label='Target', y_lims=[1], 
                      gridsize=(80,40), max_counts=30, cmap='ocean_r', n_std=4,
                      savename=f'{model_name}_predictions.png'))

# Run the testing
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTesting complete.')