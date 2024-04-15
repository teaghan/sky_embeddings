import os
import numpy as np
import configparser
import torch

from utils.misc import str2bool, parseArguments, h5_snr
from utils.vit import build_model
from utils.dataloaders import build_unions_dataloader
from utils.eval_fns import ft_predict
from utils.plotting_fns import plot_resid_hexbin, evaluate_z, plot_progress, plot_conf_mat

from sklearn.model_selection import train_test_split

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
    if os.path.exists(model_filename.replace('.pth.tar', '_best.pth.tar')):
        model_filename = model_filename.replace('.pth.tar', '_best.pth.tar')
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
    loss_fn = config['TRAINING']['loss_fn']
    if 'mse' in loss_fn.lower():
        y_lims = [(0,0.005), (0,0.1)]
    else:
        y_lims = [(0,0.2), (0.7,1)]

    # Plot training progress
    plot_progress(losses, y_lims=y_lims, 
                  savename=os.path.join(fig_dir, 
                                        f'{os.path.basename(model_filename).split(".")[0]}_progress.png'))
    
    # Data loaders
    num_workers = min([os.cpu_count(),12*n_gpu])

    dataloader = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=0.0, eval=True,
                                                img_size=int(mae_config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                label_keys=eval(mae_config['DATA']['label_keys']),
                                                eval_data_file=(config['DATA']['lp_regress_data_file_train']),
                                                augment=str2bool(config['TRAINING']['augment']))
    
    print(len(dataloader.dataset))
    train_val_idx, test_idx = train_test_split(range(len(dataloader.dataset)), test_size=0.2, random_state=42)
    #train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42) 

    dataloader_val = build_unions_dataloader(batch_size=int(config['TRAINING']['batch_size']),
                                                num_workers=num_workers,
                                                patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                                num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                                max_mask_ratio=0.0, eval=True,
                                                img_size=int(mae_config['ARCHITECTURE']['img_size']),
                                                num_patches=model.module.patch_embed.num_patches,
                                                label_keys=eval(mae_config['DATA']['label_keys']),
                                                eval_data_file=(config['DATA']['lp_regress_data_file_train']),
                                                augment=str2bool(config['TRAINING']['augment']), indices=test_idx)
    
    print('The validation set consists of %i cutouts.' % (len(dataloader_val.dataset)))

    with torch.no_grad():
        tgt_labels, pred_labels = ft_predict(model, dataloader_val, device,
                                            use_label_errs=str2bool(config['TRAINING']['use_label_errs']))

    snr_vals = h5_snr(h5_path=os.path.join(data_dir, config['DATA']['lp_regress_data_file_train']), 
                      n_central_pix=8, batch_size=5000, num_samples=None)[test_idx]
    
    # Calculate minimum snr of the 5 channels
    print(len(test_idx))
    print(snr_vals.shape)
    print(pred_labels.shape, tgt_labels.shape) # not 0.2 of 6k loaded?
    snr = np.nanmin(snr_vals[:,:5], axis=(1))
    
    # Only display objects that are not super noisy
    snr_indices = snr>5
    print(len(np.where(snr_indices)[0]))

    if 'mse' in loss_fn.lower():
        #plot_resid_hexbin([r'$Z$'], tgt_labels[snr_indices], pred_labels[snr_indices], y_lims=[1], 
        plot_resid_hexbin([r'$Z$'], tgt_labels, pred_labels, y_lims=[1], 
                          gridsize=(80,40), max_counts=5, cmap='ocean_r', n_std=4,
                          savename=os.path.join(fig_dir, f'{model_name}_predictions.png'))
        
        #evaluate_z(pred_labels[snr_indices], tgt_labels[snr_indices], n_bins=8, z_range=(0.2,1.6), threshold=0.1, 
        #           #y_lims=[(-0.08,0.08),(-0.02,0.02),(0,0.03),(0,0.1)], 
        #           snr=snr[snr_indices],
        #           savename=os.path.join(fig_dir, f'{model_name}_redshift.png'))
        evaluate_z(pred_labels, tgt_labels, n_bins=8, z_range=(0.2,1.6), threshold=0.1,
                   savename=os.path.join(fig_dir, f'{model_name}_redshift.png'))
        
    else:
        # Turn logit predictions into classes
        pred_class = np.argmax(pred_labels, 1)
        tgt_class = tgt_labels[:,0]        
        labels = ['galaxy', 'qso', 'star']
        # Plot confusion Matrix
        plot_conf_mat(tgt_class, pred_class, labels, 
                      savename=os.path.join(fig_dir, f'{model_name}_classes.png'))
        

# Run the testing
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTesting complete.')