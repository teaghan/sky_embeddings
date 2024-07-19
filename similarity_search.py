import argparse
import os
import configparser
import torch
import numpy as np
import ast

from utils.misc import str2bool, h5_snr
from utils.mim_vit import build_model as build_mim
from utils.vit import build_model as build_vit
from utils.dataloaders import build_h5_dataloader
from utils.plotting_fns import display_images, plot_dual_histogram, normalize_images
from utils.eval_fns import mae_latent
from utils.similarity import mae_simsearch, compute_similarity

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser('Similarity searching.', add_help=False)

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    parser.add_argument("-tgt_fn", "--target_fn", 
                        type=str, default='HSC_dud_dwarf_galaxy_calexp_GIRYZ7610_64.h5')
    parser.add_argument("-tst_fn", "--test_fn", 
                        type=str, default='HSC_dud_unknown_calexp_GIRYZ7610_64.h5')
    parser.add_argument("-tgt_i", "--target_indices", 
                        default='[1,2]')
    parser.add_argument("-aug", "--augment_targets", 
                        type=str, default='True')
    parser.add_argument("-mp", "--max_pool", 
                        type=str, default='True')
    parser.add_argument("-ct", "--cls_token", 
                        type=str, default='False')
    parser.add_argument("-snr", "--snr_range", 
                        default='[2,7]')
    parser.add_argument("-bs", "--batch_size", 
                        type=int, default=64)
    parser.add_argument("-m", "--metric", 
                        type=str, default='cosine')
    parser.add_argument("-c", "--combine", 
                        type=str, default='min')
    parser.add_argument("-dc", "--display_channel", 
                        type=int, default=2)
    parser.add_argument("-np", "--n_plot", 
                        type=int, default=36)
    parser.add_argument("-ns", "--n_save", 
                        type=int, default=300)
    
    # Alternative data directory than sky_embeddings/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    return parser

# Load arguments
args = parseArguments()
args = args.parse_args()
model_name = args.model_name
target_fn = args.target_fn
test_fn = args.test_fn
if args.target_indices!='None':
    target_indices = ast.literal_eval(args.target_indices)
else:
    target_indices = None
augment_targets = str2bool(args.augment_targets)
max_pool = str2bool(args.max_pool)
cls_token = str2bool(args.cls_token)
snr_range = ast.literal_eval(args.snr_range)
batch_size = args.batch_size
metric = args.metric
combine = args.combine
display_channel = args.display_channel
n_plot = args.n_plot
n_save = args.n_save

# Directories
cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
data_dir = args.data_dir
if data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')
fig_dir = os.path.join(cur_dir, 'figures/')
results_dir = os.path.join(cur_dir, 'results/')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
# Determine device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_gpu = torch.cuda.device_count()
num_workers = min([os.cpu_count(),12*n_gpu])
print(f'Using Torch version: {torch.__version__}')
print(f'Using a {device} device with {n_gpu} GPU(s)')

# Load model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')

model_filename =  os.path.join(model_dir, model_name+'.pth.tar') 

if 'pretained_mae' in config['TRAINING']:
    mae_name = config['TRAINING']['pretained_mae']
    if mae_name=='None':
        mae_filename = 'None'
        mae_config = config
    else:
        # Load pretrained MAE configuration
        mae_config = configparser.ConfigParser()
        mae_config.read(config_dir+mae_name+'.ini')
        mae_filename =  os.path.join(model_dir, mae_name+'.pth.tar')
        
    # Construct the model and load pretrained weights
    model, losses, cur_iter = build_vit(config, mae_config, 
                                        model_filename, mae_filename,
                                        device, build_optimizer=False)

else:
    mae_config = config
    # Construct the model and load pretrained weights
    model, losses, cur_iter = build_mim(config, model_filename, device, build_optimizer=False)

# Calculate S/N of images in test dataset
print('Estimating S/N for test dataset images...')
test_snr = h5_snr(os.path.join(data_dir, test_fn), n_central_pix=8, batch_size=5000)
# Calculate minimum snr of the 5 channels
#test_snr = np.min(test_snr, axis=(1))
test_snr = np.nanmin(test_snr[:,:5], axis=(1))

# Only use images in specified S/N range
test_indices = np.where((test_snr>snr_range[0]) & (test_snr<snr_range[1]))[0]

# Data loaders
target_dataloader = build_h5_dataloader(os.path.join(data_dir, target_fn), 
                                     batch_size=batch_size, 
                                     num_workers=num_workers,
                                     img_size=int(config['ARCHITECTURE']['img_size']),
                                     num_patches=model.module.patch_embed.num_patches,
                                     patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                     num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                     max_mask_ratio=None,
                                     shuffle=False,
                                     indices=target_indices)

test_dataloader = build_h5_dataloader(os.path.join(data_dir, test_fn), 
                                   batch_size=batch_size, 
                                   num_workers=num_workers,
                                   img_size=int(config['ARCHITECTURE']['img_size']),
                                   num_patches=model.module.patch_embed.num_patches,
                                   patch_size=int(mae_config['ARCHITECTURE']['patch_size']), 
                                   num_channels=int(mae_config['ARCHITECTURE']['num_channels']), 
                                   max_mask_ratio=None,
                                   shuffle=False,
                                   indices=test_indices)


# Map target samples to latent-space
target_latent, target_images = mae_latent(model, target_dataloader, device, return_images=True, 
                                          apply_augmentations=augment_targets, num_augmentations=64,
                                         remove_cls=False)

# Plot targets
display_images(normalize_images(target_images[:,display_channel,:,:].data.cpu().numpy()), 
                                vmin=0., vmax=1, savename=os.path.join(fig_dir, f'{model_name}_{target_fn[:-3]}_simsearch_target.png'))

# Compute similarity score for all test samples and select the best ones
test_images, test_latent, test_ra_decs, test_scores = mae_simsearch(model, target_latent, test_dataloader, 
                                device, metric=metric, combine=combine, use_weights=True,
                               max_pool=max_pool, cls_token=cls_token, nested_batches=False, n_save=n_save)

# Display top n_plot candidates
display_images(normalize_images(test_images[:n_plot,display_channel,:,:].data.cpu().numpy()), 
                                vmin=0., vmax=1, savename=os.path.join(fig_dir, f'{model_name}_{target_fn[:-3]}_simsearch_results_f.png'))

# Save results
np.savez(os.path.join(results_dir, f'{model_name}_{target_fn[:-3]}_simsearch_results_f.npz'), 
         test_ra_decs=test_ra_decs.data.cpu().numpy(), test_scores=test_scores.data.cpu().numpy(),
        target_images=target_images.data.cpu().numpy(), target_features=target_latent.data.cpu().numpy(), 
        test_images=test_images.data.cpu().numpy(), test_features=test_latent.data.cpu().numpy())