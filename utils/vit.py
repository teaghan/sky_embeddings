from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn

import timm.optim.optim_factory as optim_factory
import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from timm.layers import AttentionPoolLatent

import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from utils.misc import str2bool
from pos_embed import interpolate_pos_embed, crop_pos_embed
from lr_decay import param_groups_lrd
from location_encoder import LocationEncoder

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
    if 'num_classes' in config['DATA'].keys():
        num_labels = int(config['DATA']['num_classes'])
    else:
        num_labels = len(eval(config['DATA']['label_keys']))
        if str2bool(config['TRAINING']['use_label_errs']):
            num_labels = num_labels//2
    label_means = len(eval(config['DATA']['label_means']))
    label_stds = len(eval(config['DATA']['label_stds']))
    dropout = float(eval(config['ARCHITECTURE']['dropout']))
    ra_dec = str2bool(mae_config['ARCHITECTURE']['ra_dec'])

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
                         drop_rate=dropout,
                             ra_dec=ra_dec)
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
                          drop_rate=dropout,
                             ra_dec=ra_dec)
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
                         drop_rate=dropout,
                             ra_dec=ra_dec)
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
                         drop_rate=dropout,
                             ra_dec=ra_dec)
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
                         drop_rate=dropout,
                             ra_dec=ra_dec)
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
                         drop_rate=dropout,
                             ra_dec=ra_dec)
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
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                         start_factor=1.0, 
                                                         end_factor=1/final_lr_factor, 
                                                         total_iters=int(total_batch_iters))


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

        # Interpolate the position embedding matrix
        checkpoint_model = checkpoint['model']
        interpolate_pos_embed(model.module, checkpoint_model)
        
        # Load model weights
        model.module.load_state_dict(checkpoint_model)
        
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
        #crop_pos_embed(model.module, checkpoint_model)
        # Interpolate the position embedding matrix
        interpolate_pos_embed(model.module, checkpoint_model)

        # Load the pre-trained model weights
        msg = model.module.load_state_dict(checkpoint_model, strict=False)
        print(msg)
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
                 ra_dec=False,
                 **kwargs):
        # Call the superclass constructor
        super(VisionTransformer, self).__init__(**kwargs)

        # Pixel normalization values
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.ra_dec = ra_dec
        
        # Label normalization values
        self.label_means = torch.tensor(label_means)
        self.label_stds = torch.tensor(label_stds)

        self.simmim = simmim

        if self.ra_dec:
            # Mapping for Right Ascension and Dec to the Embedding space
            self.ra_dec_embed = LocationEncoder(neural_network_name="siren", 
                                                legendre_polys=5,
                                                dim_hidden=8,
                                                num_layers=1,
                                                num_classes=kwargs['embed_dim'])
            self.num_extra_tokens = 2
        else:
            self.num_extra_tokens = 1

        # Fixed sin-cos embedding to identify the spatial position of each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_extra_tokens, 
                                                  kwargs['embed_dim']), requires_grad=False)

        # Trainable pixel values that replace the masked pixels
        self.patch_mask_values = nn.Parameter(torch.zeros((kwargs['in_chans'], 
                                                           self.patch_embed.patch_size[0], 
                                                           self.patch_embed.patch_size[0])))
        
        # Pre-compute the tile size based on expected image dimensions
        self.tile_size = (kwargs['img_size'])//(self.patch_embed.patch_size[0])

        # Prediction Head
        if kwargs['global_pool'] == 'map':
            self.attn_pool = AttentionPoolLatent(
                kwargs['embed_dim'],
                num_heads=2,
                mlp_ratio=kwargs['mlp_ratio'],
                norm_layer=kwargs['norm_layer'],
            )
        else:
            self.attn_pool = None


    def norm_inputs(self, x):
        return (x - self.pixel_mean) / self.pixel_std
    
    def normalize_labels(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.label_means) / self.label_stds
    
    def denormalize_labels(self, labels):
        '''Rescale the labels back to their original units.'''
        return labels * self.label_stds + self.label_means

    def normalize_ra_dec(self, ra_dec):
        """
        Normalize RA and Dec values in a tensor to be between -1 and 1.
        
        Parameters:
        - ra_dec_tensor: A tensor of shape (batch_size, 2) where the first column contains RA and the second contains Dec,
          both in degrees.
          
        Returns:
        - A tensor of the same shape where RA and Dec values are normalized to the range [-1, 1].
        """
        # RA: Scale from [0, 360] to [-1, 1]
        normalized_ra = (ra_dec[:, 0] / 180.0) - 1.0
        
        # Dec: Scale from [-90, 90] to [-1, 1]
        normalized_dec = ra_dec[:, 1] / 90.0
        
        # Stack the normalized RA and Dec back into a tensor of the same shape
        return torch.stack((normalized_ra, normalized_dec), dim=1)

    def forward_features(self, x, ra_dec=None, mask=None, reshape_out=False):

        B, C, H, W = x.shape
        # Normalize input images
        x = self.norm_inputs(x)
        
        # Expand the masking values to match the size of the batch of images
        patch_mask_values = self.patch_mask_values.repeat(1, H//self.patch_embed.patch_size[0], 
                                                          W//self.patch_embed.patch_size[1])
        patch_mask_values = patch_mask_values.expand(B, -1, -1, -1)
        
        # Replace NaN values with patch_mask_values
        x = torch.where(torch.isnan(x), patch_mask_values, x)
        # Image is masked where mask==1 and replaced with the values in patch_mask_values
        if mask is not None:
            x = x * (1 - mask) + patch_mask_values * mask
        
        # Embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, self.num_extra_tokens:, :]

        if self.ra_dec:
            # Map RA and Dec to embedding space and add positional embedding
            ra_dec = self.ra_dec_embed(ra_dec) + self.pos_embed[:, 1]
            # Append RA and Dec token
            x = torch.cat((ra_dec.unsqueeze(1), x), dim=1)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        if reshape_out:
            x = x[:, self.num_extra_tokens:]
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x, None, None
    
    def forward(self, x: torch.Tensor, mask=None, ra_dec=None) -> torch.Tensor:
        x, _, _ = self.forward_features(x, ra_dec=ra_dec)
        x = self.forward_head(x)
        return x

def vit_base(label_means, label_stds, 
             pixel_mean, pixel_std, ra_dec, simmim=False, **kwargs):
    model = VisionTransformer(label_means, label_stds,
                              pixel_mean, pixel_std, simmim, ra_dec=ra_dec,
                              depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(label_means, label_stds, 
             pixel_mean, pixel_std,  ra_dec, **kwargs):
    model = VisionTransformer(label_means, label_stds, 
                              pixel_mean, pixel_std, ra_dec=ra_dec,
                              depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(label_means, label_stds, 
             pixel_mean, pixel_std,  ra_dec, **kwargs):
    model = VisionTransformer(label_means, label_stds, 
                              pixel_mean, pixel_std, ra_dec=ra_dec,
                              depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
