import torch
import time

def get_train_samples(dataloader, nested_batches):
    '''Accomodates both dataloaders.'''
    if nested_batches:
        # Iterate through all of the tiles
        for sample_batches, masks, ra_decs in dataloader:
            # Iterate through each batch of images in this tile of the sky
            for samples, mask, ra_dec in zip(sample_batches[0], masks[0], ra_decs[0]):
                yield samples, mask, ra_dec
    else:
        for samples, mask, ra_dec in dataloader:
            yield samples, mask, ra_dec

import torch

def update_best_scores(samples, ra_decs, similarity_scores, 
                       best_samples, best_ra_decs, best_scores, n_save, metric):
    if metric == 'cosine':
        combined_scores = torch.cat((best_scores, similarity_scores), dim=0)
        combined_samples = torch.cat((best_samples, samples), dim=0)
        combined_ra_decs = torch.cat((best_ra_decs, ra_decs), dim=0)
        sorted_indices = torch.argsort(combined_scores, descending=True)
    else:
        combined_scores = torch.cat((best_scores, similarity_scores), dim=0)
        combined_samples = torch.cat((best_samples, samples), dim=0)
        combined_ra_decs = torch.cat((best_ra_decs, ra_decs), dim=0)
        sorted_indices = torch.argsort(combined_scores, descending=False)

    best_scores = combined_scores[sorted_indices][:n_save]
    best_samples = combined_samples[sorted_indices][:n_save]
    best_ra_decs = combined_ra_decs[sorted_indices][:n_save]

    return best_samples, best_ra_decs, best_scores

def mae_simsearch(model, target_latent, dataloader, device, n_batches=None, 
                  metric='cosine', combine='min', use_weights=True, max_pool=False, 
                  cls_token=False, nested_batches=True, n_save=256, verbose=100):

    if not nested_batches:
        if n_batches is None:
            n_batches = len(dataloader)
        print(f'Performing similarity search on {min(len(dataloader), n_batches)} batches...')
    else:
        print(f'Performing similarity search on {len(dataloader)} tiles...')
    model.eval()

    if hasattr(model, 'module'):
        num_extra_tokens = model.module.num_extra_tokens
    else:
        num_extra_tokens = model.num_extra_tokens
    
    target_latent = target_latent.to(device, non_blocking=True)
    if cls_token:
        # Use cls token
        target_latent = target_latent[:,:1]
    else:
        # Remove cls token and any other extra tokens
        target_latent = target_latent[:,num_extra_tokens:]
        if max_pool:
            # Select max feature across all samples
            target_latent, _ = torch.max(target_latent, dim=1, keepdim=True)

    best_ra_decs = torch.empty((n_save, 2), device=device)
    best_scores = torch.full((n_save,), float('-inf') if metric == 'cosine' else float('inf'), device=device)

    time_start = time.time()
    with torch.no_grad():
        # Loop through spectra in dataset
        for i, (samples, masks, ra_decs) in enumerate(get_train_samples(dataloader, nested_batches)):   
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            ra_decs = ra_decs.to(device, non_blocking=True)

            if i==0:
                best_samples = torch.empty((n_save, *samples.shape[1:]), device=device)

            # Map to latent space
            if hasattr(model, 'module'):
                test_latent, _, _ = model.module.forward_features(samples, ra_dec=ra_decs, 
                                                                  reshape_out=False)
            else:
                test_latent, _, _ = model.forward_features(samples, ra_dec=ra_decs,
                                                           reshape_out=False)

            if cls_token:
                # Use cls token
                test_latent = test_latent[:,:1]
            else:
                # Remove cls token
                test_latent = test_latent[:,num_extra_tokens:]
                if max_pool:
                    # Select max feature across all samples
                    test_latent, _ = torch.max(test_latent, dim=1, keepdim=True)

            # Try to put all features on the same scale
            if i == 0:
                mean_feats = test_latent.mean(dim=(0, 1))
                std_feats = test_latent.std(dim=(0, 1), unbiased=True) 
                target_latent = (target_latent - mean_feats) / (std_feats + 1e-8)
            test_latent = (test_latent - mean_feats) / (std_feats + 1e-8)

            # Compute similarity score for each sample
            test_similarity = compute_similarity(target_latent, test_latent, 
                                                 metric=metric, combine=combine, use_weights=use_weights)
            
            best_samples, best_ra_decs, best_scores = update_best_scores(samples, ra_decs, 
                                                                         test_similarity, best_samples, 
                                                                         best_ra_decs, best_scores, n_save, metric)
            
            if not nested_batches:
                if (i+1) % verbose == 0:
                    print(f'Processed {i+1}/{n_batches} image batches...', end='\r')
                
                if (i+1) >= n_batches:
                    break
            else:
                if (i+1) % verbose == 0:
                    time_spent = time.time() - time_start # s
                    time_per_batch = time_spent/(i+1)
                    print(f'Processed {i+1} image batches ({time_per_batch:0.2f} seconds per batch)...', end='\r')

        # Map to latent space
        if hasattr(model, 'module'):
            best_latent, _, _ = model.module.forward_features(best_samples, ra_dec=best_ra_decs, 
                                                                reshape_out=False)
        else:
            best_latent, _, _ = model.forward_features(best_samples, ra_dec=best_ra_decs,
                                                        reshape_out=False)
    
    return best_samples, best_latent, best_ra_decs, best_scores

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