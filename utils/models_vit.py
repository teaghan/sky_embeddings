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
from pos_embed import interpolate_pos_embed
from pretrain import str2bool
from lr_decay import param_groups_lrd

def build_model(config, mae_config, model_filename, mae_filename, device, build_optimizer=False):

    # Model architecture
    img_size = int(config['ARCHITECTURE']['img_size'])
    num_channels = int(mae_config['ARCHITECTURE']['num_channels'])
    patch_size = int(mae_config['ARCHITECTURE']['patch_size'])
    model_type = mae_config['ARCHITECTURE']['model_type']
    num_labels = len(eval(config['DATA']['label_keys']))
    label_means = len(eval(config['DATA']['label_means']))
    label_stds = len(eval(config['DATA']['label_stds']))

    # Construct the model
    if model_type=='base':
        model = vit_base(label_means=label_means,
                         label_stds=label_stds,
                         img_size=img_size,
                         in_chans=num_channels,
                         patch_size=patch_size,
                         num_classes=num_labels)
                        #global_pool=global_pool)

    elif model_type=='large':
        model = vit_large(label_means=label_means,
                         label_stds=label_stds,
                         img_size=img_size,
                         in_chans=num_channels,
                         patch_size=patch_size,
                         num_classes=num_labels)
                        #global_pool=global_pool)
    elif model_type=='huge':
        model = vit_huge(label_means=label_means,
                         label_stds=label_stds,
                         img_size=img_size,
                         in_chans=num_channels,
                         patch_size=patch_size,
                         num_classes=num_labels)
                        #global_pool=global_pool)
    model.to(device)

    if build_optimizer:
        total_batch_iters = int(float(config['TRAINING']['total_batch_iters']))
        enc_init_lr = float(config['TRAINING']['enc_init_lr'])
        head_init_lr = float(config['TRAINING']['head_init_lr'])
        enc_weight_decay = float(config['TRAINING']['enc_weight_decay'])
        head_weight_decay = float(config['TRAINING']['head_weight_decay'])
        final_lr_factor = float(config['TRAINING']['final_lr_factor'])
        lw_lrd = str2bool(config['TRAINING']['layerwise_lr_decay'])
        layer_decay = float(config['TRAINING']['layer_decay'])
        
        if lw_lrd:
            # Build optimizer with layer-wise lr decay
            param_groups = param_groups_lrd(model, enc_weight_decay,
                                            no_weight_decay_list=model.no_weight_decay(),
                                            layer_decay=layer_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=enc_init_lr)
            max_lr = enc_init_lr
        else:
            # Separate the head parameters from the rest of the model
            head_params = model.head.parameters()
            enc_params = filter(lambda p: id(p) not in map(id, model.head.parameters()), model.parameters())
            
            # Create the optimizer with two parameter groups
            optimizer = torch.optim.AdamW([{'params': enc_params, 'lr': enc_init_lr, 'enc_weight_decay': enc_weight_decay},
                                           {'params': head_params, 'lr': head_init_lr, 'head_weight_decay': head_weight_decay}])
            max_lr = [enc_init_lr, head_init_lr]
            
        # Learning rate scheduler for the two learning rates
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
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
    

def load_model(model, model_filename, mae_filename, optimizer=None, lr_scheduler=None):
    
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
        model.load_state_dict(checkpoint['model'])
        
    else:
        print('\nLoading pre-trained MAE model weights...')

        checkpoint = torch.load(mae_filename, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        #print(msg)

        if model.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, label_means, label_stds, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Label normalization values
        self.label_means = torch.tensor(label_means)#.to(device)
        self.label_stds = torch.tensor(label_stds)#.to(device)
        
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def normalize_labels(self, labels):
        '''Normalize each label to have zero-mean and unit-variance.'''
        return (labels - self.label_means) / self.label_stds
    
    def denormalize_labels(self, labels):
        '''Rescale the labels back to their original units.'''
        return labels * self.label_stds + self.label_means
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            #x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_base(label_means, label_stds, **kwargs):
    model = VisionTransformer(label_means, label_stds,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(label_means, label_stds, **kwargs):
    model = VisionTransformer(label_means, label_stds,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(label_means, label_stds, **kwargs):
    model = VisionTransformer(label_means, label_stds,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model