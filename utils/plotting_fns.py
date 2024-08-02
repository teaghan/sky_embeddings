import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10})

def plot_progress(losses, y_lims=[(0,1)], x_lim=None, lp=False,
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize

    num_ax = 1
    if 'train_lp_acc' in losses.keys():
        num_ax += 1
    if 'train_lp_r2' in losses.keys():
        num_ax += 1
    if ('train_acc' in losses.keys()) or ('train_mae' in losses.keys()):
        num_ax += 1
        
    fig = plt.figure(figsize=(9,3*(num_ax)))
    
    gs = gridspec.GridSpec(num_ax, 1)
    
    linestyles = ['-', '--', '-.', ':']

    ax1 = plt.subplot(gs[0])
    axs = [ax1]
    cur_ax = 0
    if 'train_lp_acc' in losses.keys():
        cur_ax +=1
        ax2 = plt.subplot(gs[cur_ax])
        axs.append(ax2)
    if 'train_lp_r2' in losses.keys():
        cur_ax +=1
        ax3 = plt.subplot(gs[cur_ax])
        axs.append(ax3)
    if ('train_acc' in losses.keys()) or ('train_mae' in losses.keys()):
        cur_ax +=1
        ax4 = plt.subplot(gs[cur_ax])
        axs.append(ax4)
    
    ax1.set_title('Objective Function', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_loss'],
                 label=r'Train', c='k')
    if 'val_loss' in losses.keys():
        ax1.plot(losses['batch_iters'], losses['val_loss'],
                     label=r'Val', c='r')
        ax1.set_ylabel('Loss',fontsize=fontsize)

    if 'train_lp_acc' in losses.keys():
        ax2.set_title('Linear Probe Classification', fontsize=fontsize)
        ax2.plot(losses['batch_iters'], losses['train_lp_acc'],
                     label=r'Train', c='k')
        ax2.plot(losses['batch_iters'], losses['val_lp_acc'],
                         label=r'Val', c='r')
        ax2.set_ylabel('Accuracy',fontsize=fontsize)
    if 'train_lp_r2' in losses.keys():
        ax3.set_title('Linear Probe Regression', fontsize=fontsize)
        ax3.plot(losses['batch_iters'], losses['train_lp_r2'],
                     label=r'Train', c='k')
        ax3.plot(losses['batch_iters'], losses['val_lp_r2'],
                         label=r'Val', c='r')
        ax3.set_ylabel(r'$R^2$',fontsize=fontsize)
    if 'train_acc' in losses.keys():
        ax4.set_title('Classification Accuracy', fontsize=fontsize)
        ax4.plot(losses['batch_iters'], losses['train_acc'],
                     label=r'Train', c='k')
        ax4.plot(losses['batch_iters'], losses['val_acc'],
                         label=r'Val', c='r')
        ax4.set_ylabel(r'Acc (\%)',fontsize=fontsize)
    elif 'train_mae' in losses.keys():
        ax4.set_title('Regression Error', fontsize=fontsize)
        ax4.plot(losses['batch_iters'], losses['train_mae'],
                     label=r'Train', c='k')
        ax4.plot(losses['batch_iters'], losses['val_mae'],
                         label=r'Val', c='r')
        ax4.set_ylabel(r'MAE',fontsize=fontsize)
    
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

def plot_batch_orig_pred(orig_imgs, pred_imgs, 
                         n_samples=5, savename=None):
    
    # Normalize the batch between 0 and 1
    orig_imgs = normalize_images(np.concatenate((orig_imgs, pred_imgs)))
    b = pred_imgs.shape[0]
    pred_imgs = orig_imgs[b:b*2]
    orig_imgs = orig_imgs[:b]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 2, figsize=(5, n_samples*5/3))
    
    # Loop through the images and display each one
    for i in range(n_samples):
        if i==0:
            axes[i,0].set_title('Target', fontsize=12)
            axes[i,1].set_title('Generated', fontsize=12)
        vmin = np.min(orig_imgs[i])
        vmax = np.max(orig_imgs[i])
        axes[i,0].imshow(orig_imgs[i], vmin=vmin, vmax=vmax)
        axes[i,1].imshow(pred_imgs[i], vmin=vmin, vmax=vmax)
        for j in range(2):
            axes[i,j].axis('off')  # Hide the axes

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, facecolor='white', transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    
    else:
        plt.show()

    plt.close()

def tile_channels(image, grid_size=None):
    """
    Rearranges the channels of an image into a tiled grid on a single axis.
    
    Parameters:
    image (numpy.ndarray): The image to rearrange, expected shape: (channels, height, width).
    grid_size (tuple, optional): The grid size (rows, cols) for tiling the channels. 
                                 If None, it's automatically determined based on the number of channels.
    
    Returns:
    numpy.ndarray: The image with channels rearranged into a tiled grid.
    """
    channels, height, width = image.shape
    if grid_size is None:
        # Determine grid size if not specified
        grid_rows = int(np.ceil(np.sqrt(channels)))
        grid_cols = int(np.ceil(channels / grid_rows))
    else:
        grid_rows, grid_cols = grid_size
    
    # Initialize an array for the tiled image
    tiled_image = np.zeros((height * grid_rows, width * grid_cols))
    
    channel_index = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if channel_index >= channels:
                break  # No more channels to process
            tiled_image[row*height:(row+1)*height, col*width:(col+1)*width] = image[channel_index]
            channel_index += 1
    
    return tiled_image

def plot_batch_tiled(orig_imgs, mask_imgs, pred_imgs, n_samples=5, savename=None):
    """
    Plots original, masked, and predicted images with all channels tiled into a single axis.
    
    Parameters:
    orig_imgs, mask_imgs, pred_imgs (numpy.ndarray): Batch of images to plot.
    n_samples (int): Number of samples to plot.
    savename (str, optional): Filename to save the plot. If None, the plot is shown.
    """
    # Normalize the batch between 0 and 1
    #orig_imgs = normalize_images(np.concatenate((orig_imgs, mask_imgs, pred_imgs)))
    #b = pred_imgs.shape[0]
    #mask_imgs = orig_imgs[b:b*2]
    #pred_imgs = orig_imgs[b*2:]
    #orig_imgs = orig_imgs[:b]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples*10/3))
    
    # Loop through the samples and display each one
    for i in range(n_samples):
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12)
            axes[i, 1].set_title('Masked Input', fontsize=12)
            axes[i, 2].set_title('Reconstruction', fontsize=12)
        
        for j, img_batch in enumerate([orig_imgs, mask_imgs, pred_imgs]):
            tiled_image = tile_channels(img_batch[i])
            #if j==0:
            #    vmin, vmax = np.nanmin(tiled_image), np.nanmax(tiled_image)
            axes[i, j].imshow(tiled_image)#, vmin=vmin, vmax=vmax)#, cmap='gray')
            axes[i, j].axis('off')  # Hide the axes

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

def plot_conf_mat(tgt_class, pred_class, labels, savename):
    cm = confusion_matrix(tgt_class, pred_class)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title(f'Classifier Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    if savename is not None:
        plt.savefig(savename, facecolor='white', 
                    transparent=False, dpi=100,
                    bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
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
        ax.annotate('$\overline{x}$=%0.3f $s$=%0.3f'% (np.mean(diff), np.std(diff)),
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
    ax.annotate(f'bias={bias:.3f}, MAD={mad:.3f}, frac={frac_out:.3f}',
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
                     y_lims=y_lims[0][1], gridsize=(100,50), max_counts=5, cmap='ocean_r', n_std=3,
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

def plot_dual_histogram(data1, data2, bins=30, data1_label='Data 1', data2_label='Data 2', title='Dual Histogram', 
                        x_label='Similarity Score', y_label='Counts', xlim=None):
    """
    Plots two sets of data in a single histogram for easy comparison.

    Parameters:
    data1 (array-like): The first dataset.
    data2 (array-like): The second dataset.
    bins (int): Number of bins for the histogram.
    data1_label (str): Label for the first dataset.
    data2_label (str): Label for the second dataset.
    title (str): Title of the plot.
    """

    if xlim is None:
        bins = np.linspace(np.min([data1, data2]), np.max([data1, data2]), bins)
    else:
        bins = np.linspace(xlim[0], xlim[1], bins)
    
    # Plot histograms
    plt.hist(data1, bins=bins, alpha=0.5, label=data1_label)
    plt.hist(data2, bins=bins, alpha=0.5, label=data2_label)

    # Add legend and labels
    plt.legend()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    if xlim is not None:
        plt.xlim(*xlim)

    # Show the plot
    plt.show()