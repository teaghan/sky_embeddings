import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_fill_holes, label
from skimage.segmentation import find_boundaries

from utils.datastream_utils import band_dict_incl
from utils.helper import adjust_flux_with_zp, find_band_indices


def upscale_mask(mask, patch_size):
    upscaled_mask = mask.repeat_interleave(patch_size, dim=0)
    upscaled_mask = upscaled_mask.repeat_interleave(patch_size, dim=1)
    return upscaled_mask


def visualize_masks(images, mask_collator, masks, masks_p, patch_size, alpha_grid, patch_grid=False):
    fig, axs = plt.subplots(nrows=len(images), ncols=3, figsize=(10, len(images) * 4))
    for i, (img, mask_indices, mask_indices_p) in enumerate(zip(images, masks, masks_p)):
        if images.shape[1] == 5:
            img = cutout_rgb(cutout=img.detach().cpu().numpy(), bands=['i', 'r', 'g'], in_dict=band_dict_incl)
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
                axs[i, 0].axvline(x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 0].axhline(y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)

            for x in range(0, img.shape[0] + 1, patch_size):
                axs[i, 1].axvline(x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 1].axhline(y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)

            for x in range(0, img.shape[0] + 1, patch_size):
                axs[i, 2].axvline(x, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)
            for y in range(0, img.shape[1] + 1, patch_size):
                axs[i, 2].axhline(y, color='black', linestyle='--', lw=0.2, alpha=alpha_grid)

        full_mask = torch.zeros(mask_collator.height, mask_collator.width, dtype=torch.float32)
        for idx in mask_indices:
            row = idx // mask_collator.width
            col = idx % mask_collator.width
            full_mask[row, col] = 1

        part_masks_p = torch.zeros(
            mask_indices_p.shape[0], mask_collator.height, mask_collator.width, dtype=torch.float32
        )
        for j, part_mask in enumerate(mask_indices_p):
            for idx in part_mask:
                row = idx // mask_collator.width
                col = idx % mask_collator.width
                part_masks_p[j, row, col] = 1

        # Upscale mask to match image resolution
        full_mask = upscale_mask(full_mask, mask_collator.patch_size)

        part_masks_p = torch.stack(
            [upscale_mask(sub_mask, mask_collator.patch_size) for sub_mask in part_masks_p]
        )
        full_mask_p = torch.any(part_masks_p, dim=0)

        # Apply semi-transparent mask
        masked_img = img.copy()
        full_mask = full_mask.detach().cpu().numpy()
        alpha = 0.6  # transparency level
        if images.shape[1] == 5:
            axs[i, 1].imshow(masked_img)
        else:
            axs[i, 1].imshow(masked_img)
        axs[i, 1].imshow(np.ma.masked_where(full_mask == 1, full_mask), cmap='cool', vmin=-1, alpha=alpha)
        axs[i, 1].set_title('Final context mask')
        axs[i, 1].axis('off')

        masked_img_p = img.copy()
        full_mask_p = full_mask_p.detach().cpu().numpy()
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
    plt.show()
    return img


def cutout_rgb(cutout, bands, in_dict):
    """
    Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout (numpy.ndarray): array containing cutout data
        bands (list): list of bands to use for the RGB image
        in_dict (dict): band dictionary

    Returns:
        PIL image: image cutout
    """
    band_idx = find_band_indices(in_dict, bands)
    cutout_rgb = cutout[band_idx]
    cutout_red = cutout_rgb[2]  # R
    cutout_green = cutout_rgb[1]  # G
    cutout_blue = adjust_flux_with_zp(cutout_rgb[0], 27.0, 30.0)  # B

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
