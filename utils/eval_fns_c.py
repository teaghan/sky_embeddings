import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from dataloaders import get_augmentations

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

            masked_samples = samples.detach().clone()
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            loss, pred, mask = model(samples, mask_ratio=mask_ratio, mask=mask)
            
            if hasattr(model, 'module'):
                model = model.module

            mask[torch.isnan(masked_samples.view(masked_samples.size(0),masked_samples.size(1),-1)).all(-1)] = 1.
            
            # Return back to original scale
            pred = model.denorm_imgs(masked_samples, pred)
            
            # Reshape to image size
            pred = model.unflatten_patches(pred)
            
            # Fill in missing prediction pixels with original values
            #pred[mask==0] = samples[mask==0]
            pred = pred.data.cpu().numpy()
            
            # Originals
            samples = masked_samples.detach().clone().data.cpu().numpy()

            # Masked inputs
            masked_samples[mask==1] = torch.nan
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

def mae_latent(model, dataloader, device, mask_ratio=0., n_batches=None, return_images=False, verbose=1, 
               apply_augmentations=False, num_augmentations=16, remove_cls=True):
    
    if n_batches is None:
        n_batches = len(dataloader)
    if verbose > 0:
        print(f'Encoding {min(len(dataloader), n_batches)} batches...')
    model.eval()

    latents = []
    images = []
    
    # Conditional application of augmentations
    augmentations = get_augmentations() if apply_augmentations else None

    with torch.no_grad():
        # Loop through spectra in dataset
        for batch_idx, (samples, masks, _) in enumerate(dataloader):

            # Apply augmentations if enabled
            augmented_samples = []
            if apply_augmentations:
                for sample in samples:
                    # Add the original sample
                    augmented_samples.append(sample.unsqueeze(0))
                    # Generate augmented versions of the sample
                    for _ in range(num_augmentations):
                        augmented_sample = augmentations(sample)
                        augmented_samples.append(augmented_sample.unsqueeze(0))
                
                # Concatenate all augmented samples along the batch dimension
                samples = torch.cat(augmented_samples, dim=0)
            
            # Switch to GPU if available
            samples = samples.to(device, non_blocking=True)

            if hasattr(model, 'module'):
                latent, _, _ = model.module.forward_encoder(samples, mask_ratio, mask=None, reshape_out=False)
                if model.module.attn_pool:
                    remove_cls = False 
            else:
                latent, _, _ = model.forward_encoder(samples, mask_ratio, mask=None, reshape_out=False)
                if model.attn_pool:
                    remove_cls = False 
            if remove_cls:
                # Remove cls token
                latent = latent[:,1:]
            
            latents.append(latent.detach().cpu())
            if return_images:
                images.append(samples.detach().cpu())
            if len(latents)>=n_batches:
                break
    if return_images:
        return torch.cat(latents), torch.cat(images)
    else:
        return torch.cat(latents)

def ft_predict(model, dataloader, device, num_batches=None, return_images=False, use_label_errs=False):
    model.eval()
    
    tgt_labels = []
    pred_labels = []
    images = []

    if num_batches is None:
        num_batches = len(dataloader)

    print(f'Running predictions on {num_batches} batches...')
    
    for i, (samples, masks, labels) in enumerate(dataloader):
        
        # Switch to GPU if available
        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_label_errs:
            # Don't need label uncertainties
            num_labels = labels.size(1)//2
            labels = labels[:,:num_labels]
    
        # Run predictions
        model_outputs = model(samples, mask=masks)

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
