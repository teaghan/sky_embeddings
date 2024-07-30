import math
import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_fill_holes, label
from skimage.segmentation import find_boundaries
from sklearn.metrics import confusion_matrix

from utils.dataloaders import get_band_indices

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times'], 'font.size': 10})


def plot_progress(losses, y_lims=[(0, 1)], x_lim=None, lp=False, fontsize=18, savename=None):
    fontsize_small = 0.8 * fontsize

    num_ax = 1
    if 'train_lp_acc' in losses.keys():
        num_ax += 1
    if 'train_lp_r2' in losses.keys():
        num_ax += 1
    if ('train_acc' in losses.keys()) or ('train_mae' in losses.keys()):
        num_ax += 1

    fig = plt.figure(figsize=(9, 3 * (num_ax)))  # noqa: F841

    gs = gridspec.GridSpec(num_ax, 1)

    linestyles = ['-', '--', '-.', ':']  # noqa: F841

    ax1 = plt.subplot(gs[0])
    axs = [ax1]
    cur_ax = 0
    if 'train_lp_acc' in losses.keys():
        cur_ax += 1
        ax2 = plt.subplot(gs[cur_ax])
        axs.append(ax2)
    if 'train_lp_r2' in losses.keys():
        cur_ax += 1
        ax3 = plt.subplot(gs[cur_ax])
        axs.append(ax3)
    if ('train_acc' in losses.keys()) or ('train_mae' in losses.keys()):
        cur_ax += 1
        ax4 = plt.subplot(gs[cur_ax])
        axs.append(ax4)

    ax1.set_title('Objective Function', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_loss'], label=r'Train', c='k')
    if 'val_loss' in losses.keys():
        ax1.plot(losses['batch_iters'], losses['val_loss'], label=r'Val', c='r')
        ax1.set_ylabel('Loss', fontsize=fontsize)

    if 'train_lp_acc' in losses.keys():
        ax2.set_title('Linear Probe Classification', fontsize=fontsize)
        ax2.plot(losses['batch_iters'], losses['train_lp_acc'], label=r'Train', c='k')
        ax2.plot(losses['batch_iters'], losses['val_lp_acc'], label=r'Val', c='r')
        ax2.set_ylabel('Accuracy', fontsize=fontsize)
    if 'train_lp_r2' in losses.keys():
        ax3.set_title('Linear Probe Regression', fontsize=fontsize)
        ax3.plot(losses['batch_iters'], losses['train_lp_r2'], label=r'Train', c='k')
        ax3.plot(losses['batch_iters'], losses['val_lp_r2'], label=r'Val', c='r')
        ax3.set_ylabel(r'$R^2$', fontsize=fontsize)
    if 'train_acc' in losses.keys():
        ax4.set_title('Classification Accuracy', fontsize=fontsize)
        ax4.plot(losses['batch_iters'], losses['train_acc'], label=r'Train', c='k')
        ax4.plot(losses['batch_iters'], losses['val_acc'], label=r'Val', c='r')
        ax4.set_ylabel(r'Acc (\%)', fontsize=fontsize)
    elif 'train_mae' in losses.keys():
        ax4.set_title('Regression Error', fontsize=fontsize)
        ax4.plot(losses['batch_iters'], losses['train_mae'], label=r'Train', c='k')
        ax4.plot(losses['batch_iters'], losses['val_mae'], label=r'Val', c='r')
        ax4.set_ylabel(r'MAE', fontsize=fontsize)

    for i, ax in enumerate(axs):
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(losses['batch_iters'][0], losses['batch_iters'][-1])
        # ax.set_ylim(*y_lims[i])
        ax.set_xlabel('Batch Iterations', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.grid(True)
        ax.legend(fontsize=fontsize_small, ncol=1)

    plt.tight_layout()

    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )

    else:
        plt.show()

    plt.close()


def normalize_images(images):
    """
    Normalize each channel of an array of images to be in the range 0 to 1.

    :param images: An array of images with shape (b, h, w, c), where c is the number of channels.
    :return: Normalized array of images.
    """

    if images.ndim == 4:
        min_ = np.nanmin(images, axis=(0, 1, 2))
        max_ = np.nanmax(images, axis=(0, 1, 2))

    else:
        min_ = np.nanmin(images)
        max_ = np.nanmax(images)

    return (images - min_) / (max_ - min_)


def plot_batch(orig_imgs, mask_imgs, pred_imgs, n_samples=5, channel_index=None, savename=None):
    if channel_index is not None:
        # Select a single channel to display
        orig_imgs = orig_imgs[:, :, :, channel_index]
        mask_imgs = mask_imgs[:, :, :, channel_index]
        pred_imgs = pred_imgs[:, :, :, channel_index]

    # Normalize the batch between 0 and 1
    orig_imgs = normalize_images(np.concatenate((orig_imgs, mask_imgs, pred_imgs)))
    b = pred_imgs.shape[0]
    mask_imgs = orig_imgs[b : b * 2]
    pred_imgs = orig_imgs[b * 2 :]
    orig_imgs = orig_imgs[:b]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 3, figsize=(5, n_samples * 5 / 3))

    # Loop through the images and display each one
    for i in range(n_samples):
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12)
            axes[i, 1].set_title('Masked Input', fontsize=12)
            axes[i, 2].set_title('Reconstruction \n+ Visible', fontsize=12)
        vmin = np.min(orig_imgs[i])
        vmax = np.max(orig_imgs[i])
        axes[i, 0].imshow(orig_imgs[i], vmin=vmin, vmax=vmax)
        axes[i, 1].imshow(mask_imgs[i], vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(pred_imgs[i], vmin=vmin, vmax=vmax)
        for j in range(3):
            axes[i, j].axis('off')  # Hide the axes

    plt.tight_layout()

    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )

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
            tiled_image[row * height : (row + 1) * height, col * width : (col + 1) * width] = image[
                channel_index
            ]
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
    # orig_imgs = normalize_images(np.concatenate((orig_imgs, mask_imgs, pred_imgs)))
    # b = pred_imgs.shape[0]
    # mask_imgs = orig_imgs[b:b*2]
    # pred_imgs = orig_imgs[b*2:]
    # orig_imgs = orig_imgs[:b]

    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples * 10 / 3))

    # Loop through the samples and display each one
    for i in range(n_samples):
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12)
            axes[i, 1].set_title('Masked Input', fontsize=12)
            axes[i, 2].set_title('Reconstruction', fontsize=12)

        for j, img_batch in enumerate([orig_imgs, mask_imgs, pred_imgs]):
            tiled_image = tile_channels(img_batch[i])
            # if j==0:
            #    vmin, vmax = np.nanmin(tiled_image), np.nanmax(tiled_image)
            axes[i, j].imshow(tiled_image)  # , vmin=vmin, vmax=vmax)#, cmap='gray')
            axes[i, j].axis('off')  # Hide the axes

    plt.tight_layout()

    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )
    else:
        plt.show()

    plt.close()


def display_images(images, vmin=0.0, vmax=1.0, show_num=True, savename=None):
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
    if grid_size == 1:
        axes = [axes]
    else:
        # Flatten the array of axes
        axes = axes.flatten()

    # Normalize images
    images[images < vmin] = vmin
    images[images > vmax] = vmax
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
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )

    else:
        plt.show()


def plot_conf_mat(tgt_class, pred_class, labels, savename):
    cm = confusion_matrix(tgt_class, pred_class)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Classifier Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )
    else:
        plt.show()


def plot_resid_hexbin(
    label_keys,
    tgt_stellar_labels,
    pred_stellar_labels,
    y_lims=[2],
    gridsize=(100, 50),
    max_counts=30,
    cmap='ocean_r',
    n_std=3,
    savename=None,
):
    fig, axes = plt.subplots(len(label_keys), 1, figsize=(10, len(label_keys) * 2.5))

    if not hasattr(axes, 'len'):
        axes = [axes]

    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1)
    for i, ax in enumerate(axes):
        label_key = label_keys[i]

        # Calculate residual
        diff = pred_stellar_labels[:, i] - tgt_stellar_labels[:, i]

        # Plot
        tgts = tgt_stellar_labels[:, i]
        x_range = [
            np.max([np.min(tgts), np.median(tgts) - n_std * np.std(tgts)]),
            np.min([np.max(tgts), np.median(tgts) + n_std * np.std(tgts)]),
        ]

        hex_data = ax.hexbin(
            tgt_stellar_labels[:, i],
            diff,
            gridsize=gridsize,
            cmap=cmap,
            extent=(x_range[0], x_range[1], -y_lims[i], y_lims[i]),
            bins=None,
            vmax=max_counts,
        )

        # Annotate with statistics
        ax.annotate(
            '$\overline{x}$=%0.3f $s$=%0.3f' % (np.mean(diff), np.std(diff)),  # type: ignore
            (0.7, 0.8),
            size=15,
            xycoords='axes fraction',
            bbox=bbox_props,
        )

        # Axis params
        ax.set_xlabel('%s$_{tgt}$' % (label_key), size=15)
        ax.set_ylabel(r'%s$_{pred}$ - %s$_{tgt}$' % (label_key, label_key), size=15)
        ax.axhline(0, linewidth=2, c='black', linestyle='--')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(-y_lims[i], y_lims[i])
        ax.set_yticks([-y_lims[i], -0.5 * y_lims[i], 0, 0.5 * y_lims[i], y_lims[i]])

        ax.tick_params(labelsize=12)
        ax.grid()

    # Colorbar
    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(hex_data, cax=cbar_ax)
    cbar.set_label('Counts', size=15)

    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )
    else:
        plt.show()


def photoz_prediction_metrics(z_pred, z_true, threshold=0.15):
    resid = (z_pred - z_true) / (1 + z_true)
    bias = np.mean(resid)
    mae_bias = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - mae_bias))
    frac_out = np.sum(np.abs(resid) > threshold) / z_pred.size

    return resid, bias, mad, frac_out


def plot_z_resid(
    fig,
    ax,
    cax,
    z_true,
    resid,
    bias,
    mad,
    frac_out,
    y_lims=1,
    gridsize=(100, 50),
    max_counts=30,
    cmap='ocean_r',
    n_std=3,
    x_range=None,
    fontsize=12,
):
    if x_range is None:
        x_range = [
            np.max([np.min(z_true), np.median(z_true) - n_std * np.std(z_true)]),
            np.min([np.max(z_true), np.median(z_true) + n_std * np.std(z_true)]),
        ]

    # Plot
    hex_data = ax.hexbin(
        z_true,
        resid,
        gridsize=gridsize,
        cmap=cmap,
        extent=(x_range[0], x_range[1], -y_lims, y_lims),
        bins=None,
        vmax=max_counts,
    )

    # Annotate with statistics
    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1)
    ax.annotate(
        f'bias={bias:.3f}, MAD={mad:.3f}, frac={frac_out:.3f}',
        (0.6, 0.8),
        size=fontsize,
        xycoords='axes fraction',
        bbox=bbox_props,
    )

    # Axis params
    # ax.set_xlabel(r'$Z_{tgt}$', size=15)
    ax.set_ylabel('Normalized\nResidual', size=fontsize)
    ax.axhline(0, linewidth=1, c='black', linestyle='--')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(-y_lims, y_lims)
    ax.set_yticks([-y_lims, -0.5 * y_lims, 0, 0.5 * y_lims, y_lims])

    # ax.grid()

    # Colorbar
    cbar = fig.colorbar(hex_data, cax=cax)
    cbar.set_label('Counts', size=fontsize)


def plot_z_scatter(fig, ax, cax, z_pred, z_true, snr, snr_max=20, y_lims=1, cmap='ocean_r', fontsize=12):
    # resid = (z_pred - z_true)

    # Plot
    scatter = ax.scatter(z_true, z_pred, c=snr, cmap=cmap, s=3, vmin=0, vmax=snr_max)

    # Axis params
    ax.set_xlabel('Spectroscopic Redshift', size=fontsize)
    ax.set_ylabel('Predicted Redshift', size=fontsize)
    ax.plot([0, 2], [0, 2], linewidth=1, c='black', linestyle='--')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    # Colorbar
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('[S/N]', size=fontsize)

    ax.grid(alpha=0.2)


def z_plots(
    z_true,
    full_resid,
    full_bias,
    full_mad,
    full_frac_out,
    bin_mids,
    bin_bias,
    bin_mad,
    bin_frac_out,
    z_range,
    fontsize=12,
    y_lims=[(-0.3, 0.3), (-0.14, 0.14), (0, 0.2), (0, 0.4)],
    savename=None,
):
    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Define a GridSpec layout
    gs = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[1, 0.02], wspace=0.02)

    # Plot distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(z_true, bins=100, range=z_range)
    ax1.set_ylabel('N', size=fontsize)
    ax1.set_xlim(*z_range)

    # Plot resid
    ax2 = fig.add_subplot(gs[1, 0])  # , sharex=ax1)
    cax2 = fig.add_subplot(gs[1, 1])
    plot_z_resid(
        fig,
        ax2,
        cax2,
        z_true,
        full_resid,
        full_bias,
        full_mad,
        full_frac_out,
        y_lims=y_lims[0][1],
        gridsize=(100, 50),
        max_counts=5,
        cmap='ocean_r',
        n_std=3,
        x_range=z_range,
        fontsize=fontsize,
    )

    # Plot bias
    ax3 = fig.add_subplot(gs[2, 0])  # , sharex=ax1)
    ax3.scatter(bin_mids, bin_bias, s=10)
    ax3.plot(bin_mids, bin_bias, linestyle='--')
    ax3.set_ylim(*y_lims[1])
    ax3.axhline(0, linewidth=1, c='black', linestyle='--')
    ax3.set_ylabel('Bias', size=fontsize)

    # Plot MAD
    ax4 = fig.add_subplot(gs[3, 0])  # , sharex=ax1)
    ax4.scatter(bin_mids, bin_mad, s=10)
    ax4.plot(bin_mids, bin_mad, linestyle='--')
    ax4.set_ylim(*y_lims[2])
    ax4.set_ylabel('MAD', size=fontsize)

    # Plot frac out
    ax5 = fig.add_subplot(gs[4, 0])  # , sharex=ax1)
    ax5.scatter(bin_mids, bin_frac_out, s=10)
    ax5.plot(bin_mids, bin_frac_out, linestyle='--')
    ax5.set_ylim(*y_lims[3])
    ax5.set_ylabel('Outlier\nFraction', size=fontsize)

    x_ticks = np.array(bin_mids) - np.diff(bin_mids)[0] / 2
    x_ticks = np.append(x_ticks, x_ticks[-1] + np.diff(bin_mids)[0])
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        ax.tick_params(labelsize=10)
        ax.set_xlim(ax1.get_xlim())
        ax.set_xticks(x_ticks)
        if i < 4:
            ax.set_xticklabels([])
        else:
            # x_ticks = np.round(ax.get_xticks(),1)
            # ax.set_xticks(x_ticks)
            # ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Spectroscopic Redshift', size=fontsize)
        ax.grid(alpha=0.2)

    if savename is not None:
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )
    else:
        plt.show()


def evaluate_z(
    z_pred,
    z_true,
    n_bins=8,
    z_range=(0.2, 2),
    y_lims=[(-0.3, 0.3), (-0.14, 0.14), (0, 0.2), (0, 0.4)],
    threshold=0.15,
    snr=None,
    savename=None,
):
    # Calculate metrics on entire dataset
    full_resid, full_bias, full_mad, full_frac_out = photoz_prediction_metrics(z_pred, z_true, threshold=0.15)

    # Split z into bins and calculate metrics
    bins = np.linspace(z_range[0], z_range[1], n_bins + 1)
    bin_indices = []
    bin_mids = np.zeros((n_bins,))
    for i in range(n_bins):
        bin_indices.append(np.where((bins[i] <= z_true) & (z_true < bins[i + 1]))[0])
        bin_mids[i] = np.mean([bins[i], bins[i + 1]])

    # Each bin should have the same number of samples
    n_samples = min([len(b) for b in bin_indices])
    print(f'Using {n_samples} from each bin')
    sample_indices = [np.random.choice(np.arange(len(b)), size=n_samples, replace=False) for b in bin_indices]  # noqa: F841
    # bin_indices = [b[b_i] for b, b_i in zip(bin_indices,sample_indices)]

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
    z_plots(
        z_true,
        full_resid,
        full_bias,
        full_mad,
        full_frac_out,
        bin_mids,
        bin_bias,
        bin_mad,
        bin_frac_out,
        z_range,
        y_lims=y_lims,
        savename=savename,
    )

    if snr is not None:
        snr_plots(
            z_pred,
            z_true,
            snr,
            n_bins,
            fontsize=12,
            cmap=snr,
            threshold=threshold,
            y_lims=y_lims,
            savename=savename,
        )


def snr_plots(
    z_pred,
    z_true,
    snr,
    n_bins,
    fontsize=12,
    cmap=None,
    snr_lim=(5, 25),
    y_lims=[(-0.3, 0.3), (-0.14, 0.14), (0, 0.2), (0, 0.4)],
    threshold=0.15,
    savename=None,
):
    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Define a GridSpec layout
    gs = gridspec.GridSpec(
        6, 2, figure=fig, width_ratios=[1, 0.02], height_ratios=[1, 1, 0.2, 1, 1, 1], wspace=0.02
    )

    # Plot one-to-one
    ax1 = fig.add_subplot(gs[:2, 0])
    cax1 = fig.add_subplot(gs[:2, 1])
    plot_z_scatter(
        fig,
        ax1,
        cax1,
        z_pred,
        z_true,
        snr,
        y_lims=y_lims[0][1],
        snr_max=snr_lim[1],
        cmap='ocean_r',
        fontsize=fontsize,
    )

    # Split snr into bins and calculate metrics
    bins = np.linspace(snr_lim[0], snr_lim[1], n_bins + 1)
    bin_indices = []
    bin_mids = np.zeros((n_bins,))
    for i in range(n_bins):
        bin_indices.append(np.where((bins[i] <= snr) & (snr < bins[i + 1]))[0])
        bin_mids[i] = np.mean([bins[i], bins[i + 1]])

    # Each bin should have the same number of samples
    n_samples = min([len(b) for b in bin_indices])
    print(f'Using {n_samples} from each bin')
    sample_indices = [np.random.choice(np.arange(len(b)), size=n_samples, replace=False) for b in bin_indices]  # noqa: F841
    # bin_indices = [b[b_i] for b, b_i in zip(bin_indices,sample_indices)]

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
    ax3 = fig.add_subplot(gs[3, 0])  # , sharex=ax1)
    ax3.scatter(bin_mids, bin_bias, s=10)
    ax3.plot(bin_mids, bin_bias, linestyle='--')
    ax3.set_ylim(*y_lims[1])
    ax3.axhline(0, linewidth=1, c='black', linestyle='--')
    ax3.set_ylabel('Bias', size=fontsize)

    # Plot MAD
    ax4 = fig.add_subplot(gs[4, 0])  # , sharex=ax1)
    ax4.scatter(bin_mids, bin_mad, s=10)
    ax4.plot(bin_mids, bin_mad, linestyle='--')
    ax4.set_ylim(*y_lims[2])
    ax4.set_ylabel('MAD', size=fontsize)

    # Plot frac out
    ax5 = fig.add_subplot(gs[5, 0])  # , sharex=ax1)
    ax5.scatter(bin_mids, bin_frac_out, s=10)
    ax5.plot(bin_mids, bin_frac_out, linestyle='--')
    ax5.set_ylim(*y_lims[3])
    ax5.set_ylabel('Outlier\nFraction', size=fontsize)

    x_ticks = np.array(bin_mids) - np.diff(bin_mids)[0] / 2
    x_ticks = np.append(x_ticks, x_ticks[-1] + np.diff(bin_mids)[0])
    for i, ax in enumerate([ax3, ax4, ax5]):
        ax.tick_params(labelsize=10)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_xticks(x_ticks)
        if i < 2:
            ax.set_xticklabels([])
        else:
            # x_ticks = np.round(ax.get_xticks(),1)
            # ax.set_xticks(x_ticks)
            # ax.set_xticklabels(x_ticks)
            ax.set_xlabel('Signal-to-Noise', size=fontsize)
        ax.grid(alpha=0.2)

    if savename is not None:
        fig_dir = os.path.dirname(savename)
        savename = os.path.basename(savename).split('.')
        savename = f'{savename[0]}_snr.{savename[1]}'
        savename = os.path.join(fig_dir, savename)
        plt.savefig(
            savename, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05
        )
    else:
        plt.show()


def plot_dual_histogram(
    data1,
    data2,
    bins=30,
    data1_label='Data 1',
    data2_label='Data 2',
    title='Dual Histogram',
    x_label='Similarity Score',
    y_label='Counts',
    xlim=None,
):
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


def upscale_mask(mask, patch_size):
    upscaled_mask = mask.repeat_interleave(patch_size, dim=0)
    upscaled_mask = upscaled_mask.repeat_interleave(patch_size, dim=1)
    return upscaled_mask


def visualize_masks(
    images,
    masks_enc,
    masks_pred,
    patch_size,
    bands,
    alpha_grid=0.5,
    n_samples=5,
    patch_grid=False,
    savename=None,
    show_plot=False,
):
    images = images.detach().cpu().numpy()
    images = images[:n_samples]

    masks_enc = masks_enc.detach().cpu().numpy()
    masks_enc = masks_enc[:n_samples]

    masks_pred = masks_pred.detach().cpu().numpy()
    masks_pred = torch.stack(masks_pred)
    masks_pred = torch.unbind(masks_pred, dim=1)
    masks_pred = masks_pred[:n_samples]

    fig, axs = plt.subplots(nrows=len(images), ncols=3, figsize=(10, len(images) * 4))
    for i, (img, mask_indices_enc, mask_indices_pred) in enumerate(zip(images, masks_enc, masks_pred)):
        if images.shape[1] >= 3:
            img = cutout_rgb(cutout=img, bands=bands, bands_rgb=['I', 'R', 'G'])
            img = np.asarray(img, dtype=np.int32)
            axs[i, 0].imshow(img)
        else:
            img = img.permute(1, 2, 0).detach().cpu().numpy()  # Change CxHxW to HxWxC for plotting
            axs[i, 0].imshow(img)

        axs[i, 0].set_xlim(0, img.shape[0])
        axs[i, 0].set_ylim(img.shape[1], 0)
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        if patch_grid:
            for x in range(0, img.shape[0] + 1, patch_size):
                axs[i, 0].axvline(
                    x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 0].axhline(
                    y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed

            for x in range(0, img.shape[0] + 1, patch_size):
                axs[i, 1].axvline(
                    x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 1].axhline(
                    y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed

            for x in range(0, img.shape[0] + 1, patch_size):
                axs[i, 2].axvline(
                    x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 2].axhline(
                    y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid
                )  # Adjust color and line style if needed

        full_mask = torch.zeros(img.shape[0] // patch_size, img.shape[1] // patch_size, dtype=torch.float32)
        for idx in mask_indices_enc:
            row = idx // (img.shape[1] // patch_size)
            col = idx % (img.shape[1] // patch_size)
            full_mask[row, col] = 1

        part_masks_p = torch.zeros(
            mask_indices_pred.shape[0],
            img.shape[0] // patch_size,
            img.shape[1] // patch_size,
            dtype=torch.float32,
        )
        for j, part_mask in enumerate(mask_indices_pred):
            for idx in part_mask:
                row = idx // img.shape[1] // patch_size
                col = idx % img.shape[1] // patch_size
                part_masks_p[j, row, col] = 1

        # Upscale mask to match image resolution
        full_mask = upscale_mask(full_mask, patch_size)

        part_masks_p = torch.stack([upscale_mask(sub_mask, patch_size) for sub_mask in part_masks_p])
        full_mask_p = torch.any(part_masks_p, dim=0)

        # Apply semi-transparent mask
        masked_img = img.copy()
        full_mask = full_mask
        alpha = 0.6  # transparency level

        if images.shape[1] == 5:
            axs[i, 1].imshow(masked_img)
        else:
            axs[i, 1].imshow(masked_img)
        axs[i, 1].imshow(np.ma.masked_where(full_mask == 1, full_mask), cmap='cool', vmin=-1, alpha=alpha)
        axs[i, 1].set_title('Final context mask')
        axs[i, 1].axis('off')

        masked_img_p = img.copy()
        full_mask_p = full_mask_p
        if images.shape[1] == 5:
            axs[i, 2].imshow(masked_img_p)
        else:
            axs[i, 2].imshow(masked_img_p)
        axs[i, 2].imshow(
            np.ma.masked_where(full_mask_p == 0, full_mask_p),
            cmap='cool',
            vmin=0,
            alpha=alpha,
            interpolation='none',
        )
        all_mask_boundaries = np.zeros(full_mask_p.shape, dtype='bool')
        for k in range(part_masks_p.shape[0]):
            mask_boundaries = find_boundaries(part_masks_p[k].detach().cpu().numpy(), mode='thin')
            all_mask_boundaries |= mask_boundaries
        axs[i, 2].imshow(
            np.ma.masked_where(all_mask_boundaries == 0, all_mask_boundaries),
            cmap='cool',
            vmin=1,
            alpha=alpha,
            interpolation='none',
        )
        axs[i, 2].set_title('Target masks')
        axs[i, 2].axis('off')

    if savename is not None:
        plt.savefig(
            savename,
            bbox_inches='tight',
            dpi=300,
        )
    if show_plot:
        plt.show()
    else:
        plt.close()


def cutout_rgb(cutout, bands, bands_rgb):
    """
    Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout (numpy.ndarray): array containing cutout data
        bands (list): list of bands to use for the RGB image
        in_dict (dict): band dictionary

    Returns:
        PIL image: image cutout
    """
    band_idx = get_band_indices(bands, bands_rgb)
    cutout_rgb = cutout[band_idx]
    # Currently assumes wrong band order [G, I, R, Y, Z]
    cutout_red = cutout_rgb[1]  # R
    cutout_green = cutout_rgb[2]  # G
    cutout_blue = cutout_rgb[0]  # B

    percentile = 99.9
    saturation_percentile_threshold = 1000.0
    high_saturation_threshold = 20000.0
    interpolate_neg = False
    min_size = 1000
    percentile_red = np.nanpercentile(cutout_red, percentile)
    percentile_green = np.nanpercentile(cutout_green, percentile)
    percentile_blue = np.nanpercentile(cutout_blue, percentile)

    #     print(f'{percentile} percentile r: {percentile_r}')
    #     print(f'{percentile} percentile g: {percentile_g}')
    #     print(f'{percentile} percentile i: {percentile_i}')

    if np.any(
        np.array([percentile_red, percentile_green, percentile_blue]) > saturation_percentile_threshold
    ):
        # If any band is highly saturated choose a lower percentile target to bring out more lsb features
        if np.any(np.array([percentile_red, percentile_green, percentile_blue]) > high_saturation_threshold):
            percentile_target = 200.0
        else:
            percentile_target = 1000.0

        # Find individual saturation percentiles for each band
        percentiles = find_percentile_from_target([cutout_red, cutout_green, cutout_blue], percentile_target)
        cutout_red_desat, _ = desaturate(
            cutout_red,
            saturation_percentile=percentiles['R'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )
        cutout_green_desat, _ = desaturate(
            cutout_green,
            saturation_percentile=percentiles['G'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )
        cutout_blue_desat, _ = desaturate(
            cutout_blue,
            saturation_percentile=percentiles['B'],  # type: ignore
            interpolate_neg=interpolate_neg,
            min_size=min_size,
        )

        rgb = np.stack(
            [cutout_red_desat, cutout_green_desat, cutout_blue_desat], axis=-1
        )  # Stack data in [R, G, B] order
    else:
        rgb = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1)

    # Create RGB image
    img_linear = rgb_image(
        rgb,
        scaling_type='linear',
        stretch=0.9,
        Q=5,
        ceil_percentile=99.8,
        dtype=np.uint8,
        do_norm=True,
        gamma=0.35,
        scale_red=1.0,
        scale_green=1.0,
        scale_blue=1.0,
    )

    img_linear = Image.fromarray(img_linear)
    img_linear = img_linear.transpose(Image.FLIP_TOP_BOTTOM)  # type: ignore

    # obj_id = cutout['cfis_id'][obj_idx].decode('utf-8').replace(' ', '_')

    # if save_rgb_cutout:
    #     img_linear.save(os.path.join(save_dir, f'{obj_id}.png'))
    # if plot_rgb_cutout:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img_linear)
    #     plt.title(obj_id, fontsize=20)
    #     plt.gca().axis('off')
    #     plt.show()

    return img_linear


def rgb_image(
    img,
    scaling_type='linear',  # Default to 'asinh', can also use 'linear'
    stretch=0.5,
    Q=10.0,
    m=0.0,
    ceil_percentile=99.8,
    dtype=np.uint8,
    do_norm=True,
    gamma=0.35,
    scale_red=1.0,
    scale_green=1.0,
    scale_blue=1.0,
):
    """
    Create an RGB image from three bands of data. The bands are assumed to be in the order RGB.

    Args:
        img (numpy.ndarray): image data in three bands, (size,size,3)
        scaling_type (str, optional): scaling type, use asinh or linear. Defaults to 'linear'.
        Q (float, optional): asinh softening parameter. Defaults to 10.
        m (float, optional): intensity that should be mapped to black. Defaults to 0.
        ceil_percentile (float, optional): percentile used to normalize the data. Defaults to 99.8.
        dtype (type, optional): dtype. Defaults to np.uint8.
        do_norm (bool, optional): normalize the data. Defaults to True.
        gamma (float, optional): perform gamma correction. Defaults to 0.35.
        scale_red (float, optional): scale contribution of red band. Defaults to 1.0.
        scale_green (float, optional): scale contribution of green band. Defaults to 1.0.
        scale_blue (float, optional): scale contribution of blue band. Defaults to 1.0.
    """

    def norm(red, green, blue, scale_red, scale_green, scale_blue):
        red = red / np.nanpercentile(red, ceil_percentile)
        green = green / np.nanpercentile(green, ceil_percentile)
        blue = blue / np.nanpercentile(blue, ceil_percentile)

        max_dtype = np.iinfo(dtype).max
        red = np.clip((red * max_dtype), 0, max_dtype)
        green = np.clip((green * max_dtype), 0, max_dtype)
        blue = np.clip((blue * max_dtype), 0, max_dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image[:, :, 0] = scale_red * red  # Red
            image[:, :, 1] = scale_green * green  # Green
            image[:, :, 2] = scale_blue * blue  # Blue
        return image

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Compute average intensity before scaling choice
    i_mean = (red + green + blue) / 3.0

    length, width = green.shape
    image = np.empty([length, width, 3], dtype=dtype)

    if scaling_type == 'asinh':
        # Apply asinh scaling
        red = red * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
        green = green * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
        blue = blue * np.arcsinh(stretch * Q * (i_mean - m)) / (Q * i_mean)
    elif scaling_type == 'linear':
        # Apply linear scaling
        max_val = np.nanpercentile(i_mean, ceil_percentile)
        red = (red / max_val) * np.iinfo(dtype).max
        green = (green / max_val) * np.iinfo(dtype).max
        blue = (blue / max_val) * np.iinfo(dtype).max
    else:
        raise ValueError(f'Unknown scaling type: {scaling_type}')

    # Optionally apply gamma correction
    if gamma is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            red = np.clip((red**gamma), 0, np.iinfo(dtype).max)
            green = np.clip((green**gamma), 0, np.iinfo(dtype).max)
            blue = np.clip((blue**gamma), 0, np.iinfo(dtype).max)

    if do_norm:
        return norm(red, green, blue, scale_red, scale_green, scale_blue)
    else:
        return np.stack([red, green, blue], axis=-1)


def find_percentile_from_target(cutouts, target_value):
    """
    Determines the first percentile from 100 to 0 where the value is less than or equal to the target value

    Args:
        cutouts (list): list of numpy.ndarrays for each band in the order [i, r, g]
        target_value (float): target value to compare against

    Returns:
        dict: dictionary containing the first percentiles where values are <= target_value for each band
    """
    results = {}
    bands = ['R', 'G', 'B']  # Define band names according to the order of input arrays
    percentiles = np.arange(100, 0, -0.01)  # Creating percentiles from 100 to 0 with 0.01 steps

    for band, cutout in zip(bands, cutouts):
        # We calculate values at each percentile
        values_at_percentiles = np.nanpercentile(cutout, percentiles)
        # Check for the first value that is <= target value
        idx = np.where(values_at_percentiles <= target_value)[0]
        if idx.size > 0:
            results[band] = percentiles[idx[0]]
        else:
            results[band] = 100.0

    return results


def desaturate(image, saturation_percentile, interpolate_neg=False, min_size=10, fill_holes=True):
    """
    Desaturate saturated pixels in an image using interpolation.

    Args:
        image (numpy.ndarray): single band image data
        saturation_percentile (float): percentile to use as saturation threshold
        interpolate_neg (bool, optional): interpolate patches of negative values. Defaults to False.
        min_size (int, optional): number of pixels in a patch to perform interpolation of neg values. Defaults to 10.
        fill_holes (bool, optional): fill holes in generated saturation mask. Defaults to True.

    Returns:
        numpy.ndarray: desaturated image, mask of saturated pixels
    """
    # Assuming image is a 2D numpy array for one color band
    # Identify saturated pixels
    mask = image >= np.nanpercentile(image, saturation_percentile)
    mask = binary_dilation(mask, iterations=2)

    if interpolate_neg:
        neg_mask = image <= 0.9

        labeled_array, num_features = label(neg_mask)  # type: ignore
        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=np.float64)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                # Accumulate the upscaled feature mask
                total_feature_mask |= component_mask

        total_feature_mask = binary_dilation(total_feature_mask, iterations=1)
        mask = np.logical_or(mask, total_feature_mask)

    if fill_holes:
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        filled_padded_mask = binary_fill_holes(padded_mask)
        if filled_padded_mask is not None:
            mask = filled_padded_mask[1:-1, 1:-1]

    y, x = np.indices(image.shape)

    # Coordinates of non-saturated pixels
    x_nonsat = x[np.logical_not(mask)]
    y_nonsat = y[np.logical_not(mask)]
    values_nonsat = image[np.logical_not(mask)]

    # Coordinates of saturated pixels
    x_sat = x[mask]
    y_sat = y[mask]

    # Interpolate to find values at the positions of the saturated pixels
    interpolated_values = griddata(
        (y_nonsat.flatten(), x_nonsat.flatten()),  # points
        values_nonsat.flatten(),  # values
        (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
        method='linear',  # 'linear', 'nearest' or 'cubic'
    )
    # If any of the interpolated values are NaN, use nearest interpolation
    if np.any(np.isnan(interpolated_values)):
        interpolated_values = griddata(
            (y_nonsat.flatten(), x_nonsat.flatten()),  # points
            values_nonsat.flatten(),  # values
            (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
            method='nearest',  # 'linear', 'nearest' or 'cubic'
        )

    # Replace saturated pixels in the image
    new_image = image.copy()
    new_image[y_sat, x_sat] = interpolated_values

    return new_image, mask
