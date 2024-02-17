import argparse
import h5py
import numpy as np
import torch
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from dataloader_simmim import build_dataloader
from similarity import mae_latent

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    
    # Alternate data directory than sky_embeddings/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    return parser


def run_iter(model, samples, masks, mask_ratio, optimizer, lr_scheduler,
             losses_cp, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)
        
    # Calculate MAE loss
    loss, _, _ = model(samples, mask_ratio=mask_ratio, mask=masks)
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

def linear_probe(model, losses_cp, device, dataloader_template, combine='pool', 
                 class_data_fn='simple_classifier_data.h5', regress_data_fn='simple_regression_data.h5'):
    model.train(False)
    data_dir = os.path.join(cur_dir, '../data/')
    
    # Classifier task
    x,y = get_embeddings(os.path.join(data_dir, class_data_fn), 
                         model, device, dataloader_template,
                         y_label='class', combine=combine)
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Creating and training a classifier
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predicting the class label
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    # Evaluating the classifier
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    losses_cp['train_lp_acc'].append(float(train_accuracy))
    losses_cp['val_lp_acc'].append(float(test_accuracy))

    # Regression task
    x,y = get_embeddings(os.path.join(data_dir, regress_data_fn), 
                         model, device, dataloader_template,
                         y_label='zspec', combine=combine)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Creating and training a linear model for regression
    regressor = LinearRegression()
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

def get_embeddings(data_path, model, device, dataloader_template, y_label='class', combine='pool'):

    # Data loader
    dataloader = build_dataloader(data_path, 
                                         norm_type=dataloader_template.dataset.norm, 
                                         batch_size=64, 
                                         num_workers=dataloader_template.num_workers,
                                         img_size=dataloader_template.dataset.img_size,
                                         pos_channel=dataloader_template.dataset.pos_channel,
                                         pix_mean=dataloader_template.dataset.global_mean,
                                         pix_std=dataloader_template.dataset.global_std,
                                         num_patches=dataloader_template.dataset.num_patches,
                                         patch_size=model.module.patch_embed.patch_size[0], 
                                         num_channels=model.module.in_chans, 
                                         max_mask_ratio=None,
                                         shuffle=False)

    # Map target samples to latent-space
    latent_features = mae_latent(model, dataloader, device, verbose=0)
    latent_features = latent_features.data.cpu().numpy()

    # Collect targets
    with h5py.File(data_path, "r") as f:
        y = f[y_label][:]

    if combine=='flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    elif combine=='pool':
        x = np.max(latent_features, axis=1)
    else:
        x = np.mean(latent_features, axis=1)

    return x, y