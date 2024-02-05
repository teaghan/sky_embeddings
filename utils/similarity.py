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

def mae_latent(model, dataloader, device, mask_ratio=0., n_batches=None, return_images=False):
    
    if n_batches is None:
        n_batches = len(dataloader)
    print(f'Encoding {min(len(dataloader), n_batches)} batches...')
    model.eval()

    latents = []
    images = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for samples, _ in dataloader:
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)

            if hasattr(model, 'module'):
                latent, _, _ = model.module.forward_encoder(samples, mask_ratio)
            else:
                latent, _, _ = model.forward_encoder(samples, mask_ratio)
            # Remove cls token
            latent = latent[:,1:]
            
            latents.append(latent.detach())
            if return_images:
                images.append(samples.detach())
            if len(latents)>=n_batches:
                break
    if return_images:
        return torch.cat(latents), torch.cat(images)
    else:
        return torch.cat(latents)

def mae_simsearch(model, target_latent, dataloader, device, n_batches=None, metric='cosine', combine='min', use_weights=True):
    
    if n_batches is None:
        n_batches = len(dataloader)
    print(f'Performing similarity search on {min(len(dataloader), n_batches)} batches...')
    model.eval()

    sim_scores = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for i, (samples, _) in enumerate(dataloader):
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)

            if hasattr(model, 'module'):
                test_latent, _, _ = model.module.forward_encoder(samples, mask_ratio=0.)
            else:
                test_latent, _, _ = model.forward_encoder(samples, mask_ratio=0.)
            # Remove cls token
            test_latent = test_latent[:,1:]

            # Normalize each feature between 0 and 1
            if i==0:
                min_feats = torch.min(torch.cat((target_latent,test_latent)).view(-1,target_latent.shape[-1]), dim=0).values
                max_feats = torch.max(torch.cat((target_latent,test_latent)).view(-1,target_latent.shape[-1]), dim=0).values
                target_latent = (target_latent - min_feats) / (max_feats - min_feats)
            
            test_latent = (test_latent - min_feats) / (max_feats - min_feats)
            

            # Compute similarity score for each sample
            test_similarity = compute_similarity(target_latent, test_latent, 
                                                 metric='cosine', combine='min', use_weights=True)
            sim_scores.append(test_similarity)
            
            if len(sim_scores)>=n_batches:
                break
    
    return torch.cat(sim_scores)

def normalize_latents(*latent):
    '''Normalize each feature to have a min of 0 and max of 1.'''
    
    # Get min and max of features after combining batch and patch dimensions
    min_ = torch.min(torch.cat(latent).view(-1,latent[0].shape[-1]), dim=0).values
    max_ = torch.max(torch.cat(latent).view(-1,latent[0].shape[-1]), dim=0).values
    
    # Normalize each feature between 0 and 1
    latent = list(latent)
    for i in range(len(latent)):
        latent[i] = (latent[i] - min_) / (max_ - min_)

    return tuple(latent)

def determine_target_features(target_latent):

    # Combine batch and patch dimensions
    target_latent = target_latent.view(-1,target_latent[0].shape[-1])
    
    # Determine average value of each feature
    avg_feat = torch.mean(target_latent, dim=0)
    
    # Determine weighting of each feature (inverse variance)
    weight_feat = 1 / torch.std(target_latent, dim=0)**2
    # Normalize to have a sum of 1
    weight_feat /= torch.sum(weight_feat)
    
    return avg_feat, weight_feat

def central_indices(tensor_2d, n):
    """
    Returns the indices of the central 'n' pixels of a 2D tensor.
    
    Parameters:
    tensor_2d (Tensor): A 2D tensor from which to find central indices.
    n (int): Number of central pixels to find.

    Returns:
    Tensor: Indices of the central 'n' pixels.
    """
    # Ensure 'n' is a square number to form a square patch of pixels
    side_length = int(n ** 0.5)
    if side_length ** 2 != n:
        raise ValueError("n must be a perfect square to form a square patch of pixels.")
    
    # Calculate the center of the tensor
    center_y, center_x = tensor_2d.shape[0] // 2, tensor_2d.shape[1] // 2
    
    # Calculate start and end indices for the slice
    start_y = center_y - side_length // 2
    end_y = start_y + side_length
    start_x = center_x - side_length // 2
    end_x = start_x + side_length
    
    # Create meshgrid of indices and then flatten
    yy, xx = torch.meshgrid(torch.arange(start_y, end_y), torch.arange(start_x, end_x), indexing="ij")
    indices = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    
    return indices

def select_centre(latent, n_patches):
    '''
    Grab the central n_patches from a set of latent features.
    
    Parameters:
    latent (Tensor): A 3D tensor from which to select the centre from (b, number of original patches, n_features)
    n_patches (int): Number of central patches to find.

    Returns:
    Tensor: A 3D tensor of the central latent features (b, n_patches, n_features)
    '''
    
    total_n_patches = latent.shape[1]
    n_patches_per_side = int(total_n_patches**0.5)
    patch_indices = torch.arange(total_n_patches).reshape((n_patches_per_side, n_patches_per_side))
    
    indices = central_indices(patch_indices, n=n_patches)
    patch_indices = patch_indices[indices[:,0],indices[:,1]]
    return latent[:,patch_indices]

def weighted_cosine_similarity(target_feats, test_feats, weights, eps=1e-6):
    """
    Compute the weighted cosine similarity between two batches of latent representations.

    Parameters:
    target_feats (torch.Tensor): Average latent features from target group. Shape: (number of features,)
    test_feats (torch.Tensor): Latent representations of batch B. Shape: (batch size, number of patches, number of features)
    weights (torch.Tensor): Feature-wise weights. Shape: (number of features,)

    Returns:
    torch.Tensor: Weighted cosine similarity for each item in the batch. Shape: (batch size, number of patches)
    """

    # Compute the dot product for each item in the batch with weights
    dot_product = torch.sum(weights * target_feats * test_feats, dim=-1)

    # Compute the magnitude of the weighted vectors
    magnitude_tgt = torch.sqrt(torch.sum(weights * target_feats ** 2, dim=-1))
    magnitude_tst = torch.sqrt(torch.sum(weights * test_feats ** 2, dim=-1))

    # Compute cosine similarity
    cosine_similarity = dot_product / (magnitude_tgt * magnitude_tst + eps)

    return cosine_similarity

def weighted_MSE(target_feats, test_feats, weights):
    """
    Compute the weighted mean-square-error between two batches of latent representations.

    Parameters:
    target_feats (torch.Tensor): Average latent features from target group. Shape: (number of features,)
    test_feats (torch.Tensor): Latent representations of batch B. Shape: (batch size, number of patches, number of features)
    weights (torch.Tensor): Feature-wise weights. Shape: (number of features,)

    Returns:
    torch.Tensor: Weighted MSE for each item in the batch. Shape: (batch size, number of patches)
    """

    # Calculate the squared error
    squared_error = (target_feats - test_feats) ** 2

    # Apply weights
    weighted_squared_error = squared_error * weights / torch.sum(weights)
    return torch.mean(weighted_squared_error, dim=-1)

def weighted_MAE(target_feats, test_feats, weights):
    """
    Compute the weighted mean-absolute-error between two batches of latent representations.

    Parameters:
    target_feats (torch.Tensor): Average latent features from target group. Shape: (number of features,)
    test_feats (torch.Tensor): Latent representations of batch B. Shape: (batch size, number of patches, number of features)
    weights (torch.Tensor): Feature-wise weights. Shape: (number of features,)

    Returns:
    torch.Tensor: Weighted MAE for each item in the batch. Shape: (batch size, number of patches)
    """

    # Calculate the absolute error
    squared_error = torch.abs(target_feats - test_feats) #** 2

    # Apply weights
    weighted_squared_error = squared_error * weights / torch.sum(weights)
    return torch.mean(weighted_squared_error, dim=-1)


def compute_similarity(target_latent, test_latent, metric='MAE', combine='mean', use_weights=True, n_central_patches=None, n_top_sims=None):

    '''
    Compute the similarity between the samples in test_latent against the features in target_latent.

    Parameters:
    target_latent (torch.Tensor): Average latent features from target group. Shape: (batch size, number of patches, number of features)
    test_latent (torch.Tensor): Latent representations of batch. Shape: (batch size, number of patches, number of features)
    metric (str): Similarity metric to use; either 'MSE', 'MAE', or 'cosine'
    combine (str): Method to combine the similarity scores for the patches in each sample; either 'mean', 'max', or 'min'.
    use_weights (bool): Whether or not to use a weighted metric for the similarity scores.
    n_central_patches (int): Number of central patches in target_latent to use for the target features.
    n_top_sims (int or None): Number of highest/lowest patch similarity scores to use when combining the patches.

    Returns:
    torch.Tensor: A single similarity value for each sample in test_latent
    
    '''
    
    if (metric=='MSE') or (metric=='MAE'):
        largest = False
    elif (metric=='cosine'):
        largest = True
    
    if n_central_patches is not None:
        # Select central patch features as targets
        target_latent = select_centre(target_latent, n_central_patches)
        # Also do for test latents?
        #test_latent = select_centre(test_latent, n_central_patches)
    
    # Determine target features and feature weighting
    target_latent, feat_weights = determine_target_features(target_latent)
    if not use_weights:
        feat_weights = torch.ones_like(feat_weights)
    
    # Compute similarities
    if metric=='MAE':
        test_similarity = weighted_MAE(target_latent, test_latent, feat_weights)
    if metric=='MSE':
        test_similarity = weighted_MSE(target_latent, test_latent, feat_weights)
    if metric=='cosine':
        test_similarity = weighted_cosine_similarity(target_latent, test_latent, feat_weights)
    
    if n_top_sims is not None:
        # Collect average of the best n_top similarities
        test_similarity = torch.topk(test_similarity, k=n_top_sims, dim=1, largest=largest).values
    
    # Combine the similarities to get a single value for each sample
    if combine=='mean':
        test_similarity = torch.mean(test_similarity, dim=1)
    elif combine=='min':
        test_similarity = torch.min(test_similarity, dim=1).values
    elif combine=='max':
        test_similarity = torch.max(test_similarity, dim=1).values
    return test_similarity

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