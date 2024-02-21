import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from similarity import get_augmentations

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

                # Unpatchify the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.in_chans)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

            if model.norm_pix_loss:
                # Return back to original scale
                pred = model.denorm_imgs(samples, pred)
            
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

def mae_latent(model, dataloader, device, mask_ratio=0., n_batches=None, return_images=False, verbose=1, 
               apply_augmentations=False, num_augmentations=16):
    
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
        for batch_idx, (samples, _, _) in enumerate(dataloader):

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
                latent, _, _ = model.module.forward_encoder(samples, mask_ratio, reshape_out=False)
            else:
                latent, _, _ = model.forward_encoder(samples, mask_ratio, reshape_out=False)
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