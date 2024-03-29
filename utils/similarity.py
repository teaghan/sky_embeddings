import random
import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

# Custom brightness adjustment for images
def adjust_brightness(img, brightness_factor):
    return img * brightness_factor

# Custom transform that applies brightness adjustment with a random factor
class RandomBrightnessAdjust:
    def __init__(self, brightness_range=(0.8, 1.2)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        brightness_factor = random.uniform(*self.brightness_range)
        return adjust_brightness(img, brightness_factor)

# Custom brightness adjustment for images
def add_noise(img, noise_factor):
    return img + torch.randn_like(img) * noise_factor

# Custom transform that applies random noise
class RandomNoise:
    def __init__(self, noise_range=(0., 0.1)):
        self.noise_range = noise_range

    def __call__(self, img):
        noise_factor = random.uniform(*self.noise_range)
        return add_noise(img, noise_factor)

# Define the augmentation pipeline
def get_augmentations(img_size=64):
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        #v2.RandomRotation(degrees=(0, 360)),
        v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        RandomBrightnessAdjust(brightness_range=(0.2, 5)),
        RandomNoise(noise_range=(0., 0.1)),
        #v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        #v2.Lambda(lambda img: img + torch.randn_like(img) * 0.05),
    ])

def mae_simsearch(model, target_latent, dataloader, device, n_batches=None, metric='cosine', combine='min', use_weights=True, max_pool=False):
    
    if n_batches is None:
        n_batches = len(dataloader)
    print(f'Performing similarity search on {min(len(dataloader), n_batches)} batches...')
    model.eval()

    if max_pool:
        # Select max feature across all samples
        target_latent, _ = torch.max(target_latent, dim=1, keepdim=True)

    sim_scores = []
    with torch.no_grad():
        # Loop through spectra in dataset
        for i, (samples, _, _) in enumerate(dataloader):
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)

            if hasattr(model, 'module'):
                test_latent, _, _ = model.module.forward_encoder(samples, mask_ratio=0., reshape_out=False)
            else:
                test_latent, _, _ = model.forward_encoder(samples, mask_ratio=0., reshape_out=False)
            # Remove cls token
            test_latent = test_latent[:,1:]

            if max_pool:
                test_latent, _ = torch.max(test_latent, dim=1, keepdim=True)

            # Try to put all features on the same scale
            if i==0:
                mean_feats = test_latent.mean(dim=(0, 1))
                std_feats = test_latent.std(dim=(0, 1), unbiased=True) 
                target_latent = (target_latent - mean_feats) / (std_feats + 1e-8)
            test_latent = (test_latent - mean_feats) / (std_feats + 1e-8)

            # Compute similarity score for each sample
            test_similarity = compute_similarity(target_latent, test_latent, 
                                                 metric='cosine', combine='min', use_weights=True)
            sim_scores.append(test_similarity)
            
            if len(sim_scores)>=n_batches:
                break
    
    return torch.cat(sim_scores)

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