import os
import numpy as np
import configparser
import torch

from utils.misc import str2bool, parseArguments, h5_snr
from utils.vit import build_model
from utils.dataloaders import build_h5_dataloader
from utils.eval_fns import ft_predict
from utils.plotting_fns import plot_resid_hexbin, evaluate_z

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
    model, losses, cur_iter = build_model(config, mae_config, 
                                          model_filename, mae_filename,
                                          device, build_optimizer=False)
    
    # Data loaders
    num_workers = min([os.cpu_count(),12*n_gpu])
    if n_gpu>1:
        batch_size = int(int(config['TRAINING']['batch_size'])/n_gpu)
    else:
        batch_size = int(config['TRAINING']['batch_size'])

    dataloader_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                        batch_size=batch_size, 
                                        num_workers=num_workers,
                                        label_keys=eval(config['DATA']['label_keys']),
                                        img_size=int(config['ARCHITECTURE']['img_size']),
                                        pos_channel=str2bool(mae_config['DATA']['pos_channel']), 
                                        patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                         num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                        num_patches=model.module.patch_embed.num_patches,
                                        shuffle=False)
    
    print('The validation set consists of %i cutouts.' % (len(dataloader_val.dataset)))

    tgt_labels, pred_labels = ft_predict(model, dataloader_val, device)

    snr_vals = h5_snr(h5_path=os.path.join(data_dir, config['DATA']['val_data_file']), 
                      n_central_pix=8, batch_size=5000, num_samples=None)
    
    # Calculate minimum snr of the 5 channels
    snr = np.min(snr_vals, axis=(1))
    
    # Only display objects that are not super noisy
    snr_indices = snr>5
    
    plot_resid_hexbin([r'$Z$'], tgt_labels[snr_indices], pred_labels[snr_indices], y_lims=[1], 
                      gridsize=(80,40), max_counts=30, cmap='ocean_r', n_std=4,
                      savename=os.path.join(fig_dir, f'{model_name}_predictions.png'))
    
    evaluate_z(pred_labels[snr_indices], tgt_labels[snr_indices], n_bins=8, z_range=(0.2,1.6), threshold=0.15, 
               y_lims=[(-0.15,0.15),(-0.1,0.1),(0,0.07),(0,0.2)], snr=snr[snr_indices],
               savename=os.path.join(fig_dir, f'{model_name}_redshift.png'))

# Run the testing
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTesting complete.')