import h5py
import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from misc import select_centre
from masks import apply_masks
from tensors import repeat_interleave_batch
from data import build_h5_dataloader, get_augmentations

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def run_iter(imgs, masks_enc, masks_pred, encoder, predictor, target_encoder, 
               optimizer, lr_scheduler, wd_scheduler, momentum_scheduler, 
               cur_iter, losses_cp, mode='train'):
    if mode=='train':
        encoder.train(True)
        predictor.train(True)
    else:
        encoder.train(False)
        predictor.train(False)

    def forward_target():
        with torch.no_grad():
            h = target_encoder(imgs)
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h

    def forward_context():
        z = encoder(imgs, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        return z

    def loss_fn(z, h):
        loss = torch.nn.functional.smooth_l1_loss(z, h)
        #loss = AllReduce.apply(loss)
        if loss.numel()>1:
            # In case of multiple GPUs
            loss = loss.unsqueeze(0).mean()
        return loss

    # Step 1. Forward
    #with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
    h = forward_target()
    z = forward_context()
    loss = loss_fn(z, h)

    #  Step 2. Backward & step
    if 'train' in mode:
        
        # Update the gradients
        loss.backward()
        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Adjust learning rate and weight decay
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        
        # Save loss and metrics
        losses_cp['train_loss'].append(float(loss))

        # Step 3. momentum update of target encoder
        with torch.no_grad():
            #m = next(momentum_scheduler)
            m = momentum_scheduler['ema'][0] + cur_iter*(momentum_scheduler['ema'][1]-momentum_scheduler['ema'][0])/(momentum_scheduler['total_batch_iters'])
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(loss))

    return losses_cp

def linear_probe(encoder, losses_cp, device, dataloader_template, class_data_path=None,
                 regress_data_path=None, combine='central'):
    '''Train a quick linear probing model to evaluate the quality of the embeddings.'''
    
    encoder.train(False)
    if class_data_path:
        # Classifier task
        x,y = get_embeddings(class_data_path, 
                             encoder, device, dataloader_template,
                             y_label='class', combine=combine)
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a classifier
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000, C=1., random_state=42)
        clf.fit(X_train, y_train)
        
        # Predicting the class label
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        
        # Evaluating the classifier
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
    
        losses_cp['train_lp_acc'].append(float(train_accuracy))
        losses_cp['val_lp_acc'].append(float(test_accuracy))
    if regress_data_path:
        # Regression task
        x,y = get_embeddings(regress_data_path, 
                             encoder, device, dataloader_template,
                             y_label='zspec', combine=combine)
    
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a linear model for regression
        #regressor = LinearRegression()
        regressor = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000, random_state=42)
        regressor.fit(X_train, y_train)
        
        # Predicting the continuous values 
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        
        # Evaluating the regressor
        #mse_test = mean_squared_error(y_test, y_pred_test)
        #mse_train = mean_squared_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        losses_cp['train_lp_r2'].append(float(r2_train))
        losses_cp['val_lp_r2'].append(float(r2_test))

def get_embeddings(data_path, encoder, device, dataloader_template, y_label='class', combine='central'):

    # Data loader
    dataloader = build_h5_dataloader(data_path, 
                                          batch_size=64, 
                                          num_workers=dataloader_template.num_workers,
                                         img_size=dataloader_template.dataset.img_size,
                                         pos_channel=dataloader_template.dataset.pos_channel,
                                         collator=None,
                                          shuffle=False)

    # Map target samples to latent-space
    latent_features = predict_latent(encoder, dataloader, device, verbose=0)
    latent_features = latent_features.data.cpu().numpy()

    # Collect targets
    with h5py.File(data_path, "r") as f:
        y = f[y_label][:]

    scale = True
    if combine=='flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    elif combine=='pool':
        x = np.max(latent_features, axis=1)
    elif combine=='centralpool':
        x = select_centre(latent_features, n_patches=16)
        x = np.max(x, axis=1)
    elif combine=='central':
        x = select_centre(latent_features, n_patches=4)
        x = x.reshape(x.shape[0], -1)
    elif combine=='mean':
        x = np.mean(latent_features, axis=1)
    else:
        x = latent_features
        x = (x - np.nanmean(x)) / np.nanstd(x)
        scale = False
        
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    return x, y

def predict_latent(encoder, dataloader, device, n_batches=None, return_images=False, verbose=1, 
                   apply_augmentations=False, num_augmentations=16):
    
    if n_batches is None:
        n_batches = len(dataloader)
    if verbose > 0:
        print(f'Encoding {min(len(dataloader), n_batches)} batches...')
    encoder.eval()

    latents = []
    images = []
    
    # Conditional application of augmentations
    augmentations = get_augmentations() if apply_augmentations else None

    with torch.no_grad():
        # Loop through spectra in dataset
        for batch_idx, samples in enumerate(dataloader):

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


            # Run prediction
            latent = encoder(samples)
            latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
            # Remove cls token
            #latent = latent[:,1:]
            
            latents.append(latent.detach().cpu())
            if return_images:
                images.append(samples.detach().cpu())
            if len(latents)>=n_batches:
                break
    if return_images:
        return torch.cat(latents), torch.cat(images)
    else:
        return torch.cat(latents)