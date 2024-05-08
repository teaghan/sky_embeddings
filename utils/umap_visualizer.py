import umap
import matplotlib.pyplot as plt

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

def plot_umap_projection(latent_representation, label_value, label_name):
    # Create UMAP object
    reducer = umap.UMAP()

    # Fit UMAP to latent representations
    umap_projection = reducer.fit_transform(latent_representation)

    # Plot UMAP projection with colored points
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_projection[:, 0], umap_projection[:, 1], c=label_value, cmap='viridis')
    plt.colorbar(label=label_name)
    plt.title('UMAP Projection with ' + label_name)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('umap_' + label_name + '.png')

# combine files later but for now just do dwards?

dataloader = build_unions_dataloader(batch_size=128, 
                                                num_workers=1,
                                                patch_size=8, 
                                                num_channels=5, 
                                                max_mask_ratio=0.0, eval=True,
                                                img_size=64,
                                                num_patches=model.module.patch_embed.num_patches,
                                                label_keys=['is_dwarf'],
                                                eval_data_file='/home/a4ferrei/scratch/data/dr5_eval_set_redshift_6k_may2024.h5',
                                                augment=False)

#x,y = get_embeddings(class_data_path, 
#                             model, device, dataloader_template_class, regression=False,
#                             y_label='is_dwarf', combine=combine, remove_cls=remove_cls)

latent_representation = x  # Latent representations
label_value = y  # Label values corresponding to each latent representation
label_name = 'is_dwarf'  # Name of the label
plot_umap_projection(latent_representation, label_value, label_name)


latent_representation = x  # Latent representations
label_value = y  # Label values corresponding to each latent representation
label_name = 'zspec'  # Name of the label
plot_umap_projection(latent_representation, label_value, label_name)


# CLASS PROJECTIONS ()
# VALUE PROJECTIONS (zspec)
# CUTOUT PROJECTIONS