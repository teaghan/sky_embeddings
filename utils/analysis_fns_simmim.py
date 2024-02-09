import numpy as np
import torch
import math
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

    else:
        plt.show()

    plt.close()

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
        for samples, mask, _ in dataloader:
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            loss, pred, mask = model(samples, mask_ratio=mask_ratio, mask=mask)

            if hasattr(model, 'module'):
                model = model.module

            if not model.simmim:
                # Put patches back in order
                pred = model.unpatchify(pred)
                if (model.input_norm is not None) or model.norm_pix_loss:
                    # Return back to original scale
                    pred = model.denorm_imgs(samples, pred)

                # Unpatchify the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.in_chans)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

            pred = torch.einsum('nchw->nhwc', pred).detach()
            mask = torch.einsum('nchw->nhwc', mask).detach()
            samples = torch.einsum('nchw->nhwc', samples)

            # Fill in missing prediction pixels with original values
            pred[mask==0] = samples[mask==0]

            # Masked inputs
            masked_samples = samples.detach().clone()
            masked_samples[mask==1] = torch.nan

            samples = samples.data.cpu().numpy()
            pred = pred.data.cpu().numpy()
            masked_samples = masked_samples.data.cpu().numpy()
            
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
    
    else:
        plt.show()

    plt.close()

def display_images(images, vmin=0., vmax=1., show_num=True, savename=None):
    """
    Display a list of images in a 2D grid using matplotlib.

    :param images: A list of images (each image should be a numpy array).
    """

    # Number of images
    num_images = len(images)

    # Calculate the grid size
    grid_size = math.ceil(math.sqrt(num_images))

    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size==1:
        axes = [axes]
    else:
        # Flatten the array of axes
        axes = axes.flatten()
    
    # Normalize images
    images[images<vmin] = vmin
    images[images>vmax] = vmax
    images = normalize_images(images)
    
    # Loop through the images and display each one
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide the axes
        if show_num:
            axes[i].set_title(str(i))
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    else:
        plt.show()

def ft_predict(model, dataloader, device, num_batches=None, return_images=False):
    model.eval()
    
    tgt_labels = []
    pred_labels = []
    images = []

    if num_batches is None:
        num_batches = len(dataloader)

    print(f'Running predictions on {num_batches} batches...')
    
    for i, (samples, labels) in enumerate(dataloader):
        
        # Switch to GPU if available
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    
        # Run predictions
        model_outputs = model(samples)

        if hasattr(model, 'module'):
            model = model.module
        
        # Rescale back to original scale
        model_outputs = model.denormalize_labels(model_outputs)
    
        # Save data
        tgt_labels.append(labels.data.cpu().numpy())
        pred_labels.append(model_outputs.data.cpu().numpy())

        if return_images:
            images.append(samples.detach().data.cpu().numpy())

        if i==num_batches:
            break
    
    tgt_labels = np.concatenate(tgt_labels)
    pred_labels = np.concatenate(pred_labels)
    if return_images:
        return tgt_labels, pred_labels, np.concatenate(images)
    else:
        return tgt_labels, pred_labels

def plot_resid_hexbin(label_keys, tgt_stellar_labels, pred_stellar_labels,
                      y_lims=[2], 
                      gridsize=(100,50), max_counts=30, cmap='ocean_r', n_std=3,
                      savename=None):
    
    fig, axes = plt.subplots(len(label_keys), 1, 
                             figsize=(10, len(label_keys)*2.5))

    if not hasattr(axes, 'len'):
        axes = [axes]

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]
            
        # Calculate residual
        diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
        
        # Plot
        tgts = tgt_stellar_labels[:,i]
        x_range = [np.max([np.min(tgts), np.median(tgts)-n_std*np.std(tgts)]),
                   np.min([np.max(tgts), np.median(tgts)+n_std*np.std(tgts)])]
        
        hex_data = ax.hexbin(tgt_stellar_labels[:,i], diff, gridsize=gridsize, cmap=cmap,
                                 extent=(x_range[0], x_range[1], -y_lims[i], y_lims[i]), 
                                 bins=None, vmax=max_counts) 
        
        # Annotate with statistics
        ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.mean(diff), np.std(diff)),
                    (0.7,0.8), size=15, xycoords='axes fraction', 
                    bbox=bbox_props)
            
        # Axis params
        ax.set_xlabel('%s$_{tgt}$' % (label_key), size=15)
        ax.set_ylabel(r'%s$_{pred}$ - %s$_{tgt}$' % (label_key, label_key), size=15)
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])

        ax.tick_params(labelsize=12)
        ax.grid()
    
    # Colorbar
    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(hex_data, cax=cbar_ax)
    cbar.set_label('Counts', size=15)
            
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

def photoz_prediction_metrics(z_pred, z_true, threshold=0.15):

    resid = (z_pred - z_true) / (1 + z_true)
    bias = np.mean(resid)
    mae_bias = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - mae_bias))
    frac_out =  np.sum(np.abs(resid)>threshold) / z_pred.size

    return resid, bias, mad, frac_out

def plot_z_resid(fig, ax, cax, z_true, resid, bias, mad, frac_out, y_lims=1, 
                 gridsize=(100,50), max_counts=30, cmap='ocean_r', n_std=3, x_range=None,
                fontsize=12):
    
    if x_range is None:
        x_range = [np.max([np.min(z_true), np.median(z_true)-n_std*np.std(z_true)]),
                   np.min([np.max(z_true), np.median(z_true)+n_std*np.std(z_true)])]

    # Plot
    hex_data = ax.hexbin(z_true, resid, gridsize=gridsize, cmap=cmap,
                         extent=(x_range[0], x_range[1], -y_lims, y_lims), 
                         bins=None, vmax=max_counts) 
    
    # Annotate with statistics
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    ax.annotate(f'bias={bias:.2f}, MAD={mad:.2f}, frac={frac_out:.2f}',
                (0.6,0.8), size=fontsize, xycoords='axes fraction', 
                bbox=bbox_props)
        
    # Axis params
    #ax.set_xlabel(r'$Z_{tgt}$', size=15)
    ax.set_ylabel('Normalized\nResidual', size=fontsize)
    ax.axhline(0, linewidth=1, c='black', linestyle='--')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(-y_lims, y_lims)
    ax.set_yticks([-y_lims, -0.5*y_lims, 0, 0.5*y_lims, y_lims])
    
    #ax.grid()
    
    # Colorbar
    cbar = fig.colorbar(hex_data, cax=cax)
    cbar.set_label('Counts', size=fontsize)

def plot_z_scatter(fig, ax, cax, z_pred, z_true, snr, snr_max=20,
                   y_lims=1, cmap='ocean_r', fontsize=12):

    #resid = (z_pred - z_true)
    
    # Plot
    scatter = ax.scatter(z_true, z_pred, c=snr, cmap=cmap, s=3, vmin=0, vmax=snr_max)     
        
    # Axis params
    ax.set_xlabel('Spectroscopic Redshift', size=fontsize)
    ax.set_ylabel('Predicted Redshift', size=fontsize)
    ax.plot([0,2],[0,2], linewidth=1, c='black', linestyle='--')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
        
    # Colorbar
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('[S/N]', size=fontsize)

    ax.grid(alpha=0.2)

def z_plots(z_true, full_resid, full_bias, full_mad, full_frac_out,
            bin_mids, bin_bias, bin_mad, bin_frac_out, z_range, fontsize=12,
            y_lims=[(-0.3,0.3),(-0.14,0.14),(0,0.2),(0,0.4)],
            savename=None):

    # Create a figure
    fig = plt.figure(figsize=(10,10))
    
    # Define a GridSpec layout
    gs = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[1,0.02], wspace=0.02)
    
    # Plot distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(z_true, bins=100, range=z_range)
    ax1.set_ylabel('N', size=fontsize)
    ax1.set_xlim(*z_range)
    
    # Plot resid
    ax2 = fig.add_subplot(gs[1, 0])#, sharex=ax1)
    cax2 = fig.add_subplot(gs[1, 1])
    plot_z_resid(fig, ax2, cax2, z_true, full_resid, full_bias, full_mad, full_frac_out, 
                     y_lims=y_lims[0][1], gridsize=(100,50), max_counts=30, cmap='ocean_r', n_std=3,
                     x_range=z_range, fontsize=fontsize)

    # Plot bias
    ax3 = fig.add_subplot(gs[2, 0])#, sharex=ax1)
    ax3.scatter(bin_mids, bin_bias, s=10)
    ax3.plot(bin_mids, bin_bias, linestyle='--')
    ax3.set_ylim(*y_lims[1])
    ax3.axhline(0, linewidth=1, c='black', linestyle='--')
    ax3.set_ylabel('Bias', size=fontsize)

    # Plot MAD
    ax4 = fig.add_subplot(gs[3, 0])#, sharex=ax1)
    ax4.scatter(bin_mids, bin_mad, s=10)
    ax4.plot(bin_mids, bin_mad, linestyle='--')
    ax4.set_ylim(*y_lims[2])
    ax4.set_ylabel('MAD', size=fontsize)

    # Plot frac out
    ax5 = fig.add_subplot(gs[4, 0])#, sharex=ax1)
    ax5.scatter(bin_mids, bin_frac_out, s=10)
    ax5.plot(bin_mids, bin_frac_out, linestyle='--')
    ax5.set_ylim(*y_lims[3])
    ax5.set_ylabel('Outlier\nFraction', size=fontsize)

    x_ticks = np.array(bin_mids) - np.diff(bin_mids)[0]/2
    x_ticks = np.append(x_ticks, x_ticks[-1] + np.diff(bin_mids)[0])
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        ax.tick_params(labelsize=10)
        ax.set_xlim(ax1.get_xlim())
        ax.set_xticks(x_ticks)
        if i<4:
            ax.set_xticklabels([])
        else:
            #x_ticks = np.round(ax.get_xticks(),1)
            #ax.set_xticks(x_ticks)
            #ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Spectroscopic Redshift', size=fontsize)
        ax.grid(alpha=0.2)
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

def snr_plots(z_pred, z_true, snr, n_bins, fontsize=12, cmap=None, snr_lim=(5,25),
              y_lims=[(-0.3,0.3),(-0.14,0.14),(0,0.2),(0,0.4)], threshold=0.15, savename=None):

    # Create a figure
    fig = plt.figure(figsize=(10,10))
    
    # Define a GridSpec layout
    gs = gridspec.GridSpec(6, 2, figure=fig, width_ratios=[1,0.02],height_ratios=[1,1,0.2,1,1,1], wspace=0.02)
        
    # Plot one-to-one
    ax1 = fig.add_subplot(gs[:2, 0])
    cax1 = fig.add_subplot(gs[:2, 1])
    plot_z_scatter(fig, ax1, cax1, z_pred, z_true, snr, 
                   y_lims=y_lims[0][1], snr_max=snr_lim[1], cmap='ocean_r', fontsize=fontsize)

    # Split snr into bins and calculate metrics
    bins = np.linspace(snr_lim[0], snr_lim[1], n_bins+1)
    bin_indices = []
    bin_mids = np.zeros((n_bins,))
    for i in range(n_bins):
        bin_indices.append(np.where((bins[i]<=snr) & (snr<bins[i+1]))[0])
        bin_mids[i] = np.mean([bins[i], bins[i+1]])

    # Each bin should have the same number of samples
    n_samples = min([len(b) for b in bin_indices])
    print(f'Using {n_samples} from each bin')
    sample_indices = [np.random.choice(np.arange(len(b)), size=n_samples, replace=False) for b in bin_indices]
    #bin_indices = [b[b_i] for b, b_i in zip(bin_indices,sample_indices)]
    
    # Calculate metrics for each bin
    bin_bias = np.zeros((n_bins,))
    bin_mad = np.zeros((n_bins,))
    bin_frac_out = np.zeros((n_bins,))
    for i, b in enumerate(bin_indices):
        resid, bias, mad, frac_out = photoz_prediction_metrics(z_pred[b], z_true[b], threshold=threshold)
        bin_bias[i] = bias
        bin_mad[i] = mad
        bin_frac_out[i] = frac_out

    # Plot bias
    ax3 = fig.add_subplot(gs[3, 0])#, sharex=ax1)
    ax3.scatter(bin_mids, bin_bias, s=10)
    ax3.plot(bin_mids, bin_bias, linestyle='--')
    ax3.set_ylim(*y_lims[1])
    ax3.axhline(0, linewidth=1, c='black', linestyle='--')
    ax3.set_ylabel('Bias', size=fontsize)

    # Plot MAD
    ax4 = fig.add_subplot(gs[4, 0])#, sharex=ax1)
    ax4.scatter(bin_mids, bin_mad, s=10)
    ax4.plot(bin_mids, bin_mad, linestyle='--')
    ax4.set_ylim(*y_lims[2])
    ax4.set_ylabel('MAD', size=fontsize)

    # Plot frac out
    ax5 = fig.add_subplot(gs[5, 0])#, sharex=ax1)
    ax5.scatter(bin_mids, bin_frac_out, s=10)
    ax5.plot(bin_mids, bin_frac_out, linestyle='--')
    ax5.set_ylim(*y_lims[3])
    ax5.set_ylabel('Outlier\nFraction', size=fontsize)

    x_ticks = np.array(bin_mids) - np.diff(bin_mids)[0]/2
    x_ticks = np.append(x_ticks, x_ticks[-1] + np.diff(bin_mids)[0])
    for i, ax in enumerate([ax3, ax4, ax5]):
        ax.tick_params(labelsize=10)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_xticks(x_ticks)
        if i<2:
            ax.set_xticklabels([])
        else:
            #x_ticks = np.round(ax.get_xticks(),1)
            #ax.set_xticks(x_ticks)
            #ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Signal-to-Noise', size=fontsize)
        ax.grid(alpha=0.2)
    
    if savename is not None:
        fig_dir = os.path.dirname(savename)
        savename = os.path.basename(savename).split('.')
        savename = f'{savename[0]}_snr.{savename[1]}'
        savename = os.path.join(fig_dir, savename)
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

def evaluate_z(z_pred, z_true, n_bins=8, z_range=(0.2,2),
               y_lims=[(-0.3,0.3),(-0.14,0.14),(0,0.2),(0,0.4)], threshold=0.15, 
               snr=None, savename=None):

    # Calculate metrics on entire dataset
    full_resid, full_bias, full_mad, full_frac_out = photoz_prediction_metrics(z_pred, z_true, threshold=0.15)
    
    # Split z into bins and calculate metrics
    bins = np.linspace(z_range[0], z_range[1], n_bins+1)
    bin_indices = []
    bin_mids = np.zeros((n_bins,))
    for i in range(n_bins):
        bin_indices.append(np.where((bins[i]<=z_true) & (z_true<bins[i+1]))[0])
        bin_mids[i] = np.mean([bins[i], bins[i+1]])
    
    # Each bin should have the same number of samples
    n_samples = min([len(b) for b in bin_indices])
    print(f'Using {n_samples} from each bin')
    sample_indices = [np.random.choice(np.arange(len(b)), size=n_samples, replace=False) for b in bin_indices]
    #bin_indices = [b[b_i] for b, b_i in zip(bin_indices,sample_indices)]
    
    # Calculate metrics for each bin
    bin_bias = np.zeros((n_bins,))
    bin_mad = np.zeros((n_bins,))
    bin_frac_out = np.zeros((n_bins,))
    for i, b in enumerate(bin_indices):
        resid, bias, mad, frac_out = photoz_prediction_metrics(z_pred[b], z_true[b], threshold=threshold)
        bin_bias[i] = bias
        bin_mad[i] = mad
        bin_frac_out[i] = frac_out

    # Create metric plot
    z_plots(z_true, full_resid, full_bias, full_mad, full_frac_out,
            bin_mids, bin_bias, bin_mad, bin_frac_out, z_range,
            y_lims=y_lims,
           savename=savename)

    if snr is not None:
        snr_plots(z_pred, z_true, snr, n_bins, fontsize=12, cmap=snr, threshold=threshold,
                  y_lims=y_lims, savename=savename)

def calculate_snr(images, n_central_pix):
    """
    Calculate the channel-wise Signal-to-Noise Ratio (SNR) for a batch of images.

    This function computes the SNR by measuring the ratio of the mean of the central 
    `n_central_pix` x `n_central_pix` pixels to the standard deviation of the 
    surrounding pixels for each channel in each image.

    Parameters:
    images (numpy.ndarray): A 4D NumPy array with shape (batch_size, n_channels, img_size, img_size).
                            This array represents the batch of images.
    n_central_pix (int): The size of the square central region from which the mean will be calculated.

    Returns:
    numpy.ndarray: A 2D NumPy array with shape (batch_size, n_channels) containing the SNR values for
                   each channel of each image in the batch.

    Note:
    - The function assumes that `n_central_pix` is smaller than `img_size`.
    - A small value (1e-8) is added to the standard deviation to avoid division by zero.
    """
    batch_size, n_channels, img_size, _ = images.shape

    # Calculate the start and end indices for the central region
    start = (img_size - n_central_pix) // 2
    end = start + n_central_pix

    # Extracting the central region for all images and channels
    central_region = images[:, :, start:end, start:end]

    # Create a mask for the surrounding region
    mask = np.ones((img_size, img_size), dtype=bool)
    mask[start:end, start:end] = False

    # Apply the mask to all images to extract the surrounding region
    surrounding_region = images[:, :, mask].reshape(batch_size, n_channels, -1)

    # Calculate the mean of the central region and the standard deviation of the surrounding region
    mean_central = np.mean(central_region, axis=(2, 3))
    std_surrounding = np.std(surrounding_region, axis=2)

    # Calculate the SNR
    snr = mean_central / (std_surrounding + 1e-8)  # Added a small value to avoid division by zero

    return snr

def h5_snr(h5_path, n_central_pix=8, batch_size=5000, num_samples=None):
        
    snr_vals = []
    with h5py.File(h5_path, "r") as f:   
        if num_samples is None:
            num_samples = len(f['cutouts'])
        if num_samples<batch_size:
            cutouts = f['cutouts'][:num_samples]
            snr_vals.append(calculate_snr(cutouts, n_central_pix))
        else:
            for i in range(0, num_samples, batch_size):
                end = min([num_samples, i+batch_size])
                cutouts = f['cutouts'][i:end]
                snr_vals.append(calculate_snr(cutouts, n_central_pix))
    
    return np.concatenate(snr_vals)