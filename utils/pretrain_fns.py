import h5py
import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from dataloaders import build_h5_dataloader
from eval_fns import mae_latent
from misc import select_centre
from utils.plotting_fns import plot_batch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

def run_iter(model, samples, ra_decs, masks, mask_ratio, optimizer, lr_scheduler,
             losses_cp, mode='train', save_sample=True):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)

    #if save_sample:
    #    plot_batch()
        
    # Run predictions and calculate loss
    loss, _, _ = model(samples, ra_dec=ra_decs, mask_ratio=mask_ratio, mask=masks)
    if loss.numel()>1:
        # In case of multiple GPUs
        loss = loss.unsqueeze(0).mean()
    
    if 'train' in mode:
        
        # Update the gradients
        loss.backward()
        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Adjust learning rate
        lr_scheduler.step()
        
        # Save loss and metrics
        losses_cp['train_loss'].append(float(loss))

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(loss))
                
    return model, optimizer, lr_scheduler, losses_cp

def linear_probe(model, losses_cp, device, dataloader_template_reg, dataloader_template_class, class_data_path=None,
                 regress_data_path=None, combine='central', remove_cls=True):
    '''Train a quick linear probing model to evaluate the quality of the embeddings.'''

    if combine=='token':
        remove_cls = False
    
    model.train(False)
    if class_data_path:
        # Classifier task
        x,y = get_embeddings(class_data_path, 
                             model, device, dataloader_template_class,
                             y_label='dwarf', combine=combine, remove_cls=remove_cls)
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a classifier
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000, C=0.01, random_state=42)
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
                             model, device, dataloader_template_reg,
                             y_label='zspec', combine=combine, remove_cls=remove_cls)
        
        # remove entries where y is NaN (because that means we don't have zspec)
        # make validation set of just known zspec ones?
        unknown_y = np.where(np.isnan(y))[0] 
        print(f'removing {len(unknown_y)} examples from linear probe set due to unknown zspec')
        x = np.delete(x, unknown_y, axis=0)
        y = np.delete(y, unknown_y, axis=0)

        unknown_x = np.where(np.isnan(x))[0] 
        print(f'removing {len(unknown_x)} examples from linear probe set due to nan in representation')
        x = np.delete(x, unknown_x, axis=0)
        y = np.delete(y, unknown_x, axis=0)

        #indices = np.where((y > 0.1) & (y < 2)) # too small of number
        #print(f'removing {len(y)-len(indices)} examples where zspec is out of range')
        #x = x[indices]
        #y = y[indices]
    
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a linear model for regression
        #regressor = LinearRegression()
        regressor = ElasticNet(alpha=0.0001, l1_ratio=0.9, max_iter=10000, random_state=42)
        regressor.fit(X_train, y_train)
        
        # Predicting the continuous values 
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)

        # TEMP
        fig = plt.figure()
        plt.scatter(y_test, y_pred_test, alpha=0.1)
        print('max(y_test):', max(y_test))
        line = np.linspace(0,max(y_test), num=5)
        plt.plot(line, line, '--')
        plt.xlabel('true zspec')
        plt.ylabel('predicted zspec')
        fig.savefig('/home/a4ferrei/scratch/github/sky_embeddings/figures/zspec_predictions.png')
        
        # Evaluating the regressor
        #mse_test = mean_squared_error(y_test, y_pred_test)
        #mse_train = mean_squared_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        losses_cp['train_lp_r2'].append(float(r2_train))
        losses_cp['val_lp_r2'].append(float(r2_test))

def get_embeddings(data_path, model, device, 
                   dataloader_template, y_label='class', combine='central', remove_cls=True, new_loader=False):

    if new_loader:
        # Data loader
        dataloader = build_h5_dataloader(data_path, 
                                            batch_size=64, 
                                            num_workers=dataloader_template.num_workers,
                                            img_size=dataloader_template.dataset.img_size,
                                            num_patches=dataloader_template.dataset.num_patches,
                                            patch_size=model.module.patch_embed.patch_size[0], 
                                            num_channels=model.module.in_chans, 
                                            max_mask_ratio=None,
                                            shuffle=False)
        
    else:
        dataloader = dataloader_template

    # Map target samples to latent-space
    latent_features = mae_latent(model, dataloader, device, verbose=0, remove_cls=remove_cls)
    latent_features = latent_features.data.cpu().numpy()

    # Collect targets
    with h5py.File(data_path, "r") as f:
        y = f[y_label][:] 

    if model.module.attn_pool:
        # There is only one output set of features if there is an attention pooling layer
        combine='flatten'

    scale = True
    if combine=='token':
        x = latent_features[:,:1].reshape(latent_features.shape[0], -1)
    elif combine=='flatten':
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
