import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
from string import ascii_lowercase

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

def plot_progress(losses, y_lims=[(0,1)], x_lim=None, lp=False,
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = 1
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    ax1 = plt.subplot(gs[0])
    
    axs = [ax1]
    
    ax1.set_title('Objective Function', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_loss'],
                 label=r'Train', c='k')
    if 'val_loss' in losses.keys():
        ax1.plot(losses['batch_iters'], losses['val_loss'],
                     label=r'Val', c='r')
    
    for i, ax in enumerate(axs):
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        ax.set_ylim(*y_lims[i])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)
        ax.legend(fontsize=fontsize_small, ncol=1)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def mae_predict(model, dataloader, device, mask_ratio, single_batch=True):
    if not single_batch:
        print('Predicting on %i batches...' % (len(dataloader)))
    model.eval()

    pred_imgs = []
    mask_imgs = []
    orig_imgs = []
    latents = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for samples, _ in dataloader:
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)

            loss, pred, mask = model(samples, mask_ratio=mask_ratio)

            pred = model.unpatchify(pred)
            pred = torch.einsum('nchw->nhwc', pred).detach()

            # Unpatchify the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.in_chans)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach()

            samples = torch.einsum('nchw->nhwc', samples)

            # Fill in missing prediction pixels with original values
            pred[mask==0] = samples[mask==0]

            # Masked inputs
            masked_samples = np.copy(samples)
            masked_samples[mask==1] = np.nan

            samples = samples.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
            
            # Save results
            pred_imgs.append(pred)
            mask_imgs.append(masked_samples)
            orig_imgs.append(samples)
            
            if single_batch:
                break
        pred_imgs = np.concatenate(pred_imgs)
        mask_imgs = np.concatenate(mask_imgs)
        orig_imgs = np.concatenate(orig_imgs)
        
    return pred_imgs, mask_imgs, orig_imgs

def normalize_images(images):
    """
    Normalize each channel of an array of images to be in the range 0 to 1.

    :param images: An array of images with shape (b, h, w, c), where c is the number of channels.
    :return: Normalized array of images.
    """
    
    if images.ndim==4:
        min_ = np.nanmin(images, axis=(0,1,2))
        max_ = np.nanmax(images, axis=(0,1,2))
        
    else:
        min_ = np.nanmin(images)
        max_ = np.nanmax(images)
            
    return (images - min_) / (max_ - min_)

def plot_batch(orig_imgs, mask_imgs, pred_imgs, 
               n_samples=5, channel_index=None, savename=None):
    
    if channel_index is not None:
        # Select a single channel to display
        orig_imgs = orig_imgs[:,:,:,channel_index]
        mask_imgs = mask_imgs[:,:,:,channel_index]
        pred_imgs = pred_imgs[:,:,:,channel_index]
    
    # Normalize the batch between 0 and 1
    orig_imgs = normalize_images(np.concatenate((orig_imgs, mask_imgs, pred_imgs)))
    b = pred_imgs.shape[0]
    mask_imgs = orig_imgs[b:b*2]
    pred_imgs = orig_imgs[b*2:]
    orig_imgs = orig_imgs[:b]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 3, figsize=(5, n_samples*5/3))
    
    # Loop through the images and display each one
    for i in range(n_samples):
        if i==0:
            axes[i,0].set_title('Original', fontsize=12)
            axes[i,1].set_title('Masked Input', fontsize=12)
            axes[i,2].set_title('Reconstruction \n+ Visible', fontsize=12)
        vmin = np.min(orig_imgs[i])
        vmax = np.max(orig_imgs[i])
        axes[i,0].imshow(orig_imgs[i], vmin=vmin, vmax=vmax)
        axes[i,1].imshow(mask_imgs[i], vmin=vmin, vmax=vmax)
        axes[i,2].imshow(pred_imgs[i], vmin=vmin, vmax=vmax)
        for j in range(3):
            axes[i,j].axis('off')  # Hide the axes

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    plt.show()