import argparse
import os
import configparser
import torch
import numpy as np
import h5py
import ast

from utils.pretrain_simmim import str2bool
from utils.models_simmim import build_model
from utils.dataloader_simmim import build_dataloader
from utils.similarity import mae_latent, select_centre

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Positional mandatory arguments
    parser.add_argument("model_names", help="Names of models to test.")

    # Optional arguments
    parser.add_argument("-cfn", "--class_data_fn", 
                        type=str, default='simple_classifier_data.h5')
    parser.add_argument("-rfn", "--regress_data_fn", 
                        type=str, default='simple_regression_data.h5')
    parser.add_argument("-c", "--combine", 
                        type=str, default='pool')
    
    # Alternative data directory than sky_embeddings/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    return parser

def get_embeddings(data_path, config, model, device, y_label='class', combine='pool'):
    # Data loader stuff
    num_workers = min([os.cpu_count(),12*n_gpu])
    if config['DATA']['norm_type']=='global':
        pix_mean = float(config['DATA']['pix_mean'])
        pix_std = float(config['DATA']['pix_std'])
    else:
        pix_mean = None
        pix_std = None
    # Data loader
    dataloader = build_dataloader(data_path, 
                                         norm_type=config['DATA']['norm_type'], 
                                         batch_size=64, 
                                         num_workers=num_workers,
                                         img_size=int(config['ARCHITECTURE']['img_size']),
                                         pos_channel=str2bool(config['DATA']['pos_channel']),
                                         pix_mean=pix_mean,
                                         pix_std=pix_std,
                                         num_patches=model.module.patch_embed.num_patches,
                                         patch_size=int(config['ARCHITECTURE']['patch_size']), 
                                         num_channels=int(config['ARCHITECTURE']['num_channels']), 
                                         max_mask_ratio=None,
                                         shuffle=False)

    # Map target samples to latent-space
    latent_features = mae_latent(model, dataloader, device)
    latent_features = latent_features.data.cpu().numpy()

    # Collect targets
    with h5py.File(data_path, "r") as f:
        y = f[y_label][:]

    if combine=='flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    if combine=='central':
        x = select_centre(latent_features, n_patches=16)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
    elif combine=='pool':
        x = np.max(latent_features, axis=1)
    else:
        x = np.mean(latent_features, axis=1)

    return x, y

if __name__=="__main__":
    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_gpu = torch.cuda.device_count()
    print(f'Using Torch version: {torch.__version__}')
    print(f'Using a {device} device with {n_gpu} GPU(s)')
    
    args = parseArguments()
    args = args.parse_args()
    
    model_names = ast.literal_eval(args.model_names)
    acc_scores = []
    regress_scores = []
    for model_name in model_names:
        
        # Directories
        cur_dir = os.path.dirname(__file__)
        config_dir = os.path.join(cur_dir, 'configs/')
        model_dir = os.path.join(cur_dir, 'models/')
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = os.path.join(cur_dir, 'data/')
        results_dir = os.path.join(cur_dir, 'results/')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        # Load model configuration
        config = configparser.ConfigParser()
        config.read(config_dir+model_name+'.ini')
        
        # Construct the model and load pretrained weights
        model_filename =  os.path.join(model_dir, model_name+'.pth.tar') 
        model, losses, cur_iter = build_model(config, model_filename, device, build_optimizer=False)

        # Classifier task
        class_data_fn = args.class_data_fn
        combine = args.combine
        x,y = get_embeddings(os.path.join(data_dir, class_data_fn), 
                             config, model, device, y_label='class', combine=combine)
        
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
        print(f'{model_name}:') 
        comment = config['Notes']['comment']
        print(f'\t {comment}')
        print(f'\t Train accuracy: {train_accuracy:.4f}')
        print(f'\t Test accuracy: {test_accuracy:.4f}')
        acc_scores.append(test_accuracy)

        # Regression task
        regress_data_fn = args.regress_data_fn
        x,y = get_embeddings(os.path.join(data_dir, regress_data_fn), 
                             config, model, device, y_label='zspec', combine=combine)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Creating and training a linear model for regression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        # Predicting the continuous values 
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        
        # Evaluating the regressor
        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        print(f'\t Train MSE: {mse_train:.4f}')
        print(f'\t Test MSE: {mse_test:.4f}')
        print(f'\t Train R2: {r2_train:.4f}')
        print(f'\t Train R2: {r2_test:.4f}')
        regress_scores.append(mse_test)
    print(f'Best classifier model: {model_names[np.argmax(acc_scores)]}')
    print(f'Best regression model: {model_names[np.argmin(regress_scores)]}')


    # Rank models for accuracy and regression scores
    acc_ranked = sorted(zip(model_names, acc_scores), key=lambda x: x[1], reverse=True)
    regress_ranked = sorted(zip(model_names, regress_scores), key=lambda x: x[1])

    # Create dictionaries to easily map model names to their rankings
    acc_rank_dict = {model: rank+1 for rank, (model, score) in enumerate(acc_ranked)}
    regress_rank_dict = {model: rank+1 for rank, (model, score) in enumerate(regress_ranked)}

    # Print the combined ranking table
    print(f"{'Model Name':<10} | {'Accuracy Ranking':<17} | {'Regression Ranking':<17} |")
    for model in model_names:
        print(f"{model:<10} | {acc_rank_dict[model]:<17} | {regress_rank_dict[model]:<17} |")
