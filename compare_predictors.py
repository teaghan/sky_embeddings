import os
import numpy as np
import configparser
import torch

from utils.misc import str2bool, parseArguments, h5_snr
from utils.vit import build_model
from utils.dataloaders import build_h5_dataloader
from utils.eval_fns import ft_predict
from utils.plotting_fns import photoz_prediction_metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, ScalarFormatter, LogFormatter, FuncFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

# Category names and model names in corresponding order
categories = ['Fully Supervised', 'Fine-tuning', 'Attentive Probing', 'Fine-tuning (Wide)', 'Fine-tuning (Wide+Large)']
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3', '#e41a1c']
#num_samples = np.array([2.5e2, 5e2, 1e3, 2e3, 4e3, 8e3, 16e3]).astype(int)
num_samples = (2**np.arange(7,15)).astype(int)
model_names = [['cls_fs_012k', 'cls_fs_05k', 'cls_fs_1k', 'cls_fs_2k', 'cls_fs_4k', 'cls_fs_8k', 'cls_fs_16k'],
               ['cls_ft_012k', 'cls_ft_025k', 'cls_ft_05k', 'cls_ft_1k', 'cls_ft_2k', 'cls_ft_4k', 'cls_ft_8k', 'cls_ft_16k'],
               ['cls_ap_012k', 'cls_ap_025k', 'cls_ap_05k', 'cls_ap_1k', 'cls_ap_2k', 'cls_ap_4k', 'cls_ap_8k', 'cls_ap_16k'],
               ['cls_ft_012k_wide', 'cls_ft_025k_wide', 'cls_ft_05k_wide', 'cls_ft_1k_wide', 'cls_ft_2k_wide', 'cls_ft_4k_wide', 'cls_ft_8k_wide', 'cls_ft_16k_wide'],
               ['cls_ft_012k_large', 'cls_ft_025k_large', 'cls_ft_05k_large', 'cls_ft_1k_large', 'cls_ft_2k_large', 'cls_ft_4k_large', 'cls_ft_8k_large', 'cls_ft_16k_large']]


# Name of data file to be used to calculate metric
val_data_file = 'HSC_dud_classes_calexp_GIRYZ7610_64_val.h5'

def metrics_vs_n(num_samples, metrics, categories, colors, fontsize=12,
                 y_lims=[(-0.14,0.14),(0,0.2),(0,0.4)], savename=None):

    # Create a figure
    fig = plt.figure(figsize=(10,4))
    
    # Define a GridSpec layout
    gs = gridspec.GridSpec(3, 1, figure=fig)

    # Plot bias
    ax3 = fig.add_subplot(gs[0, 0])
    ax3.set_ylim(*y_lims[0])
    ax3.axhline(0, linewidth=1, c='black', linestyle='--')
    ax3.set_ylabel('Bias', size=fontsize)

    # Plot MAD
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_ylim(*y_lims[1])
    ax4.set_ylabel('MAD', size=fontsize)

    # Plot frac out
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_ylim(*y_lims[2])
    #ax5.set_ylabel('Outlier\nFraction', size=fontsize)
    #ax5.set_ylabel('MAE', size=fontsize)
    ax5.set_ylabel('MSE', size=fontsize)

    handles, labels = [], []  # Initialize empty lists for handles and labels

    for i, ax in enumerate([ax3,ax4,ax5]):
        for j, label in enumerate(categories):
            scatter = ax.scatter(num_samples, metrics[j,i], s=10, c=colors[j], label=label)
            ax.plot(num_samples, metrics[j,i], linestyle='--', c=colors[j])
            if i == 0:  # Only add labels and handles from the first set of plots to avoid duplicates
                handles.append(scatter)
                labels.append(label)
        
        ax.set_xticks(num_samples)        
        ax.tick_params(labelsize=10)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Number of Training Samples', size=fontsize)
        ax.grid(alpha=0.2)

    if len(categories)>=3:
        ncol = 3
    else:
        ncol = len(categories)
    fig.legend(handles, labels, loc='upper center', 
               fontsize=fontsize,
               ncol=ncol, bbox_to_anchor=(0.5, 1.))

    plt.subplots_adjust(top=0.87)

    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

def accuracy_vs_n(num_samples, accuracies, categories, colors, fontsize=12, y_lims=[(0, 1)], savename=None):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Set the limits for y-axis
    ax.set_ylim(*y_lims[0])
    
    # Set labels
    ax.set_ylabel('Accuracy', size=fontsize)
    ax.set_xlabel('Number of Training Samples', size=fontsize)

    # Plot each category
    for j, label in enumerate(categories):
        ax.scatter(num_samples, accuracies[j], s=10, c=colors[j], label=label)
        ax.plot(num_samples, accuracies[j], linestyle='--', c=colors[j])

    # Customize ticks and grids
    #ax.set_xticks(num_samples)
    #ax.tick_params(labelsize=10)
    #ax.tick_params(labelsize=10, axis='x', rotation=90)
    #ax.grid(alpha=0.2)

    # Customize ticks and grids
    #ax.set_xscale('log', base=2)
    #ax.xaxis.set_major_locator(LogLocator(base=2.0))
    #ax.xaxis.set_major_formatter(LogFormatter(base=2, labelOnlyBase=True))
    #ax.grid(alpha=0.2)

    # Customize ticks and grids
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    
    def log_formatter(x, pos):
        return f"$2^{{{int(np.log2(x))}}}$"
    
    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.grid(alpha=0.2)

    # Add legend
    if len(categories)>=3:
        ncol = 3
        top = 0.8
    else:
        ncol = len(categories)
        top = 1.0
    fig.legend(loc='upper center', 
               fontsize=fontsize,
               ncol=ncol, bbox_to_anchor=(0.5, 1.))

    # Adjust layout
    plt.subplots_adjust(top=top)

    # Save or show the plot
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()


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

    # Calculate metrics for all models
    scores = np.zeros((len(categories), 3, len(num_samples)))
    for i in range(len(categories)):
        for j, model_name in enumerate(model_names[i]):
            # Load model configuration
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
                
            # Load pretrained model weights
            model, losses, cur_iter = build_model(config, mae_config, 
                                                  model_filename, mae_filename,
                                                  device, build_optimizer=False)
            loss_fn = config['TRAINING']['loss_fn']
            
            # Data loader
            num_workers = min([os.cpu_count(),12*n_gpu])
            dataloader_val = build_h5_dataloader(os.path.join(data_dir, config['DATA']['val_data_file']), 
                                                batch_size=int(config['TRAINING']['batch_size']), 
                                                num_workers=num_workers,
                                                label_keys=eval(config['DATA']['label_keys']),
                                                img_size=int(config['ARCHITECTURE']['img_size']),
                                                patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                                 num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                                num_patches=model.module.patch_embed.num_patches,
                                                shuffle=False)
            
            print('The validation set consists of %i cutouts.' % (len(dataloader_val.dataset)))
        
            with torch.no_grad():
                tgt_labels, pred_labels = ft_predict(model, dataloader_val, device,
                                                    use_label_errs=str2bool(config['TRAINING']['use_label_errs']))

            if 'mse' in loss_fn.lower():
                # Calculate metrics for redshift estimations
                resid, bias, mad, frac_out = photoz_prediction_metrics(pred_labels, tgt_labels, threshold=0.15)
                scores[i,0,j] = bias
                scores[i,1,j] = mad
                #scores[i,2,j] = frac_out
                #scores[i,2,j] = np.mean(np.abs(tgt_labels-pred_labels))
                scores[i,2,j] = np.mean((tgt_labels-pred_labels)**2)
            else:
                # Accuracy for classifier model
                # Turn logit predictions into classes
                pred_class = np.argmax(pred_labels, 1)
                tgt_class = tgt_labels[:,0]        
                labels = ['galaxy', 'qso', 'star']
                # Compute accuracy
                acc = np.mean(pred_class == tgt_class)

                scores[i,0,j] = acc

    if 'mse' in loss_fn.lower():
        metrics_vs_n(num_samples, scores, categories, colors,
                     y_lims=[(-0.01,0.01),(0,0.025),(0,0.01)], fontsize=14,
                     savename=os.path.join(fig_dir, f'numsamples_redshift.png'))
    else:
        accuracy_vs_n(num_samples, scores[:,0,:], categories, colors,
                     y_lims=[(0.5,1.0)], fontsize=14,
                     savename=os.path.join(fig_dir, f'numsamples_class.png'))

# Run the testing
if __name__=="__main__":
    args = parseArguments()
    args = args.parse_args()
    main(args)
    
    print('\nTesting complete.')
