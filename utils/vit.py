from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn

import timm.optim.optim_factory as optim_factory
import timm.models.vision_transformer
from timm.models.layers import trunc_normal_

import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from pos_embed import interpolate_pos_embed, crop_pos_embed
from lr_decay import param_groups_lrd

def build_model(config, mae_config, model_filename, mae_filename, device, build_optimizer=False):

    # Model architecture
    img_size = int(config['ARCHITECTURE']['img_size'])
    pixel_mean = float(mae_config['ARCHITECTURE']['pixel_mean'])
    pixel_std = float(mae_config['ARCHITECTURE']['pixel_std'])
    num_channels = int(mae_config['ARCHITECTURE']['num_channels'])
    embed_dim = int(mae_config['ARCHITECTURE']['embed_dim'])
    patch_size = int(mae_config['ARCHITECTURE']['patch_size'])
    model_type = mae_config['ARCHITECTURE']['model_type']
    global_pool = config['ARCHITECTURE']['global_pool']
    num_labels = len(eval(config['DATA']['label_keys']))
    label_means = len(eval(config['DATA']['label_means']))
    label_stds = len(eval(config['DATA']['label_stds']))
    dropout = float(eval(config['ARCHITECTURE']['dropout']))

    # Construct the model
    if model_type=='base':
        model = vit_base(label_means=label_means,
                         label_stds=label_stds,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         img_size=img_size,
                         in_chans=num_channels,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         num_classes=num_labels,
                         global_pool=global_pool,
                         drop_rate=dropout)
    elif model_type=='large':
        model = vit_large(label_means=label_means,
                          label_stds=label_stds,
                          pixel_mean=pixel_mean,
                          pixel_std=pixel_std,
                          img_size=img_size,
                          in_chans=num_channels,
                          embed_dim=embed_dim,
                          patch_size=patch_size,
                          num_classes=num_labels,
                          global_pool=global_pool,
                          drop_rate=dropout)
    elif model_type=='huge':
        model = vit_huge(label_means=label_means,
                         label_stds=label_stds,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         img_size=img_size,
                         in_chans=num_channels,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         num_classes=num_labels,
                         global_pool=global_pool,
                         drop_rate=dropout)
    elif model_type=='simmim':
        model = vit_base(label_means=label_means,
                         label_stds=label_stds,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         simmim=True,
                         img_size=img_size,
                         in_chans=num_channels,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         num_classes=num_labels,
                         global_pool=global_pool,
                         drop_rate=dropout)
    elif model_type=='mimlarge':
        model = vit_large(label_means=label_means,
                         label_stds=label_stds,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         simmim=True,
                         img_size=img_size,
                         in_chans=num_channels,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         num_classes=num_labels,
                         global_pool=global_pool,
                         drop_rate=dropout)
    elif model_type=='mimhuge':
        model = vit_huge(label_means=label_means,
                         label_stds=label_stds,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         simmim=True,
                         img_size=img_size,
                         in_chans=num_channels,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         num_classes=num_labels,
                         global_pool=global_pool,
                         drop_rate=dropout)
    model.to(device)

    # Use multiple GPUs if available
    model = nn.DataParallel(model)

    if build_optimizer:
        total_batch_iters = int(float(config['TRAINING']['total_batch_iters']))
        init_lr = float(config['TRAINING']['init_lr'])
        weight_decay = float(config['TRAINING']['weight_decay'])
        final_lr_factor = float(config['TRAINING']['final_lr_factor'])
        train_method = config['TRAINING']['train_method']
        layer_decay = float(config['TRAINING']['layer_decay'])
        
        if train_method=='finetune' or train_method=='ft':
            print('\nUsing the fine-tuning training method...')
            # Build optimizer with layer-wise lr decay
            param_groups, init_lr = param_groups_lrd(model.module, weight_decay,
                                                     no_weight_decay_list=model.module.no_weight_decay(),
                                                     layer_decay=layer_decay)
            optimizer = torch.optim.AdamW(param_groups)
            
        elif train_method=='linearprobe' or train_method=='lp':
            print('\nUsing the linear probing training method...')
            # Only train the head parameters of the model
            components_to_train = [model.module.norm, model.module.fc_norm, model.module.head]
            if global_pool=='map':
                components_to_train.append(model.module.attn_pool)

            param_groups = [{'params': m.parameters()} for m in components_to_train]
            optimizer = torch.optim.AdamW(param_groups, lr=init_lr, weight_decay=weight_decay)

            # Freeze all other parameters
            for param in model.module.parameters():
                param.requires_grad = False
            for component in components_to_train:
                for param in component.parameters():
                    param.requires_grad = True

        else:
            print('\nUsing the fully supervised training method...')
            # Train all model parameters equally
            
            # Set weight decay to 0 for bias and norm layers
            param_groups = optim_factory.param_groups_weight_decay(model, weight_decay)
    
            # Optimizer
            optimizer = torch.optim.AdamW(param_groups, lr=init_lr)
            
        # Learning rate scheduler for the two learning rates
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=init_lr,
                                                           total_steps=int(total_batch_iters), 
                                                           pct_start=0.05, anneal_strategy='cos', 
                                                           cycle_momentum=True, 
                                                           base_momentum=0.85, 
                                                           max_momentum=0.95, div_factor=25.0, 
                                                           final_div_factor=final_lr_factor, 
                                                           three_phase=False)

        # Load the model weights
        model, losses, cur_iter = load_model(model, model_filename, mae_filename, optimizer, lr_scheduler)
        
        return model, losses, cur_iter, optimizer, lr_scheduler
    else:
        # Load the model weights
        model, losses, cur_iter = load_model(model, model_filename, mae_filename)
        return model, losses, cur_iter


def load_model(model, model_filename, mae_filename='None', optimizer=None, lr_scheduler=None):
    
    # Check for pre-trained weights
    if os.path.exists(model_filename):
        # Load saved model state
        print('\nLoading saved model weights...')
        
        # Load model info
        checkpoint = torch.load(model_filename, 
                                map_location=lambda storage, loc: storage)
        losses = defaultdict(list, dict(checkpoint['losses']))
        cur_iter = checkpoint['batch_iters']+1

        # Load optimizer states
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # Load model weights
        model.module.load_state_dict(checkpoint['model'])
        
    elif mae_filename!='None':
        print('\nLoading pre-trained MAE model weights...')

        checkpoint = torch.load(mae_filename, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.module.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Crop the central positional embedding matrix
        crop_pos_embed(model.module, checkpoint_model)
        # Interpolate the position embedding matrix
        #interpolate_pos_embed(model, checkpoint_model)

        # Load the pre-trained model weights
        msg = model.module.load_state_dict(checkpoint_model, strict=False)
        #print(msg)
        #assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        
        # Manually initialize the head layer weights
        trunc_normal_(model.module.head.weight, std=2e-5)
        
        losses = defaultdict(list)
        cur_iter = 1

    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with the option of an additional input norm layer."""
    
    def __init__(self, label_means, label_stds, pixel_mean, pixel_std, simmim=False,
                 **kwargs):
        # Call the superclass constructor
        super(VisionTransformer, self).__init__(**kwargs)

        # Pixel normalization values
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        
        # Label normalization values
        self.label_means = torch.tensor(label_means)
        self.label_stds = torch.tensor(label_stds)

        self.simmim = simmim
        if self.simmim:
            # Trainable pixel values that replace the masked pixels
            self.mask_token = nn.Parameter(torch.zeros((kwargs['in_chans'], 
                                                        self.patch_embed.patch_size[0], 
                                                        self.patch_embed.patch_size[0])))

    def norm_inputs(self, x):
        return (x - self.pixel_mean) / self.pixel_std
    
    def normalize_labels(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.label_means) / self.label_stds
    
    def denormalize_labels(self, labels):
        '''Rescale the labels back to their original units.'''
        return labels * self.label_stds + self.label_means
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        if self.simmim:
            # Mask input image
            B, C, H, W = x.shape

            # Compute the tile size based on expected image dimensions
            tile_size = (H // self.mask_token.shape[1], W // self.mask_token.shape[2])
            # Expand the masking values to match the size of the batch of images
            patch_mask_values = self.mask_token.repeat(1, tile_size[0], tile_size[1])
            patch_mask_values = patch_mask_values.expand(B, -1, -1, -1)
            
            # Image is masked where mask==1 and replaced with the values in patch_mask_values
            # Additionally, replace NaN values with patch_mask_values
            x = torch.where(torch.isnan(x), patch_mask_values, x)
            
            if mask is not None:
                x = x * (1 - mask) + patch_mask_values * mask
        
        x = self.norm_inputs(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def vit_base(label_means, label_stds, 
             pixel_mean, pixel_std, simmim=False, **kwargs):
    model = VisionTransformer(label_means, label_stds,
                              pixel_mean, pixel_std, simmim,
                              depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(label_means, label_stds, 
             pixel_mean, pixel_std,  **kwargs):
    model = VisionTransformer(label_means, label_stds, 
                              pixel_mean, pixel_std,
                              depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(label_means, label_stds, 
             pixel_mean, pixel_std,  **kwargs):
    model = VisionTransformer(label_means, label_stds, 
                              pixel_mean, pixel_std,
                              depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
