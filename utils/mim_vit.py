from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import timm.optim.optim_factory as optim_factory

from timm.models.vision_transformer import PatchEmbed, Block

import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from pos_embed import get_2d_sincos_pos_embed
from misc import str2bool

def build_model(config, model_filename, device, build_optimizer=False):

    # Model architecture
    norm_pix_loss = str2bool(config['TRAINING']['norm_pix_loss'])
    img_size = int(config['ARCHITECTURE']['img_size'])
    pixel_mean = float(config['ARCHITECTURE']['pixel_mean'])
    pixel_std = float(config['ARCHITECTURE']['pixel_std'])
    num_channels = int(config['ARCHITECTURE']['num_channels'])
    embed_dim = int(config['ARCHITECTURE']['embed_dim'])
    patch_size = int(config['ARCHITECTURE']['patch_size'])
    model_type = config['ARCHITECTURE']['model_type']
    loss_fn = config['TRAINING']['loss_fn']

    # Construct the model
    if model_type=='base':
        model = mae_vit_base(embed_dim=embed_dim, 
                             img_size=img_size,
                             in_chans=num_channels,
                             patch_size=patch_size,
                             norm_pix_loss=norm_pix_loss,
                             loss_fn=loss_fn,
                             pixel_mean=pixel_mean,
                             pixel_std=pixel_std)
    elif model_type=='large':
        model = mae_vit_large(embed_dim=embed_dim,
                              img_size=img_size,
                              in_chans=num_channels,
                              patch_size=patch_size,
                              norm_pix_loss=norm_pix_loss,
                              loss_fn=loss_fn,
                              pixel_mean=pixel_mean,
                              pixel_std=pixel_std)
    elif model_type=='huge':
        model = mae_vit_huge(embed_dim=embed_dim,
                             img_size=img_size,
                             in_chans=num_channels,
                             patch_size=patch_size,
                             norm_pix_loss=norm_pix_loss,
                             loss_fn=loss_fn,
                             pixel_mean=pixel_mean,
                             pixel_std=pixel_std)
    elif model_type=='simmim':
        model = simmim_vit(embed_dim=embed_dim,
                           img_size=img_size,
                           in_chans=num_channels,
                           patch_size=patch_size,
                           norm_pix_loss=norm_pix_loss,
                           simmim=True,
                           loss_fn=loss_fn,
                           pixel_mean=pixel_mean,
                           pixel_std=pixel_std)
    elif model_type=='mimlarge':
        model = mim_vit_large(embed_dim=embed_dim,
                           img_size=img_size,
                           in_chans=num_channels,
                           patch_size=patch_size,
                           norm_pix_loss=norm_pix_loss,
                           simmim=True,
                           loss_fn=loss_fn,
                           pixel_mean=pixel_mean,
                           pixel_std=pixel_std)
    elif model_type=='mimhuge':
        model = mim_vit_huge(embed_dim=embed_dim,
                           img_size=img_size,
                           in_chans=num_channels,
                           patch_size=patch_size,
                           norm_pix_loss=norm_pix_loss,
                           simmim=True,
                           loss_fn=loss_fn,
                           pixel_mean=pixel_mean,
                           pixel_std=pixel_std)
    model.to(device)

    # Use multiple GPUs if available
    model = nn.DataParallel(model)

    if build_optimizer:
        total_batch_iters = int(float(config['TRAINING']['total_batch_iters']))
        weight_decay = float(config['TRAINING']['weight_decay'])
        init_lr = float(config['TRAINING']['init_lr'])
        final_lr_factor = float(config['TRAINING']['final_lr_factor'])
        
        # Set weight decay to 0 for bias and norm layers
        param_groups = optim_factory.param_groups_weight_decay(model, weight_decay)

        # Optimizer
        optimizer = torch.optim.AdamW(param_groups, lr=init_lr, betas=(0.9, 0.95))

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr,
                                                           total_steps=int(total_batch_iters), 
                                                           pct_start=0.05, anneal_strategy='cos', 
                                                           cycle_momentum=True, 
                                                           base_momentum=0.85, 
                                                           max_momentum=0.95, div_factor=25.0, 
                                                           final_div_factor=final_lr_factor, 
                                                           three_phase=False)
        model, losses, cur_iter = load_model(model, model_filename, optimizer, lr_scheduler)
        
        return model, losses, cur_iter, optimizer, lr_scheduler
    else:
        model, losses, cur_iter = load_model(model, model_filename)
        return model, losses, cur_iter


def load_model(model, model_filename, optimizer=None, lr_scheduler=None):
    
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
        
    else:
        print('\nStarting fresh model to train...')
        losses = defaultdict(list)
        cur_iter = 1
        
    return model, losses, cur_iter

class MaskedAutoencoderViT(nn.Module):
    '''Masked Autoencoder with VisionTransformer backbone.'''
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 simmim=False, loss_fn='mse', pixel_mean=0, pixel_std=1.):
        super().__init__()

        self.simmim = simmim
        self.loss_fn = loss_fn
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        if self.simmim:
            # Trainable pixel values that replace the masked pixels
            self.mask_token = nn.Parameter(torch.zeros((in_chans, patch_size, patch_size)))

            # Pre-compute the tile size based on expected image dimensions
            self.tile_size = (img_size)//(patch_size)
            
            # Pre-compute patch_mask_values for the expected image dimensions
            #self.patch_mask_values = self.mask_token.repeat(1, self.tile_size, self.tile_size)
            
            # Simple decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=embed_dim,
                    out_channels=self.tile_size ** 2 * in_chans, kernel_size=1),
                nn.PixelShuffle(self.tile_size),
            )

        else:
            # --------------------------------------------------------------------------
            # MAE decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
    
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
    
            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if not self.simmim:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
            
    def forward_encoder(self, x, mask_ratio=0, mask=None, reshape_out=True):
        if self.simmim:
            ids_restore = None
            if mask is not None:
                # Mask input image
                B, C, H, W = x.shape

                # Expand the masking values to match the size of the batch of images
                patch_mask_values = self.mask_token.repeat(1, self.tile_size, self.tile_size)
                patch_mask_values = patch_mask_values.expand(B, -1, -1, -1)
                
                # Image is masked where mask==1 and replaced with the values in patch_mask_values
                # Additionally, replace NaN values with patch_mask_values
                x = torch.where(torch.isnan(x), patch_mask_values, x)
                x = x * (1 - mask) + patch_mask_values * mask
        
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if not self.simmim:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.simmim and reshape_out:
            x = x[:, 1:]
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        if not self.simmim:
            # embed tokens
            x = self.decoder_embed(x)
    
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    
            # add pos embed
            x = x + self.decoder_pos_embed
    
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
    
            # predictor projection
            x = self.decoder_pred(x)
    
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = self.decoder(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # Invert nan_mask because we want 1s where the values are NOT NaN (valid for loss calculation)
        valid_data_mask = ~torch.isnan(imgs)
        valid_data_mask = valid_data_mask.to(imgs.dtype)

        # Combine the valid data mask with the existing mask to exclude both NaN values and unseen pixels
        mask = valid_data_mask * mask
        
        if self.simmim:
            if self.norm_pix_loss:
                imgs = self.patchify(imgs)
                # Compute mean and variance of patches in target
                mean, var = patch_mean_and_var(imgs)
                #mean = imgs.mean(dim=-1, keepdim=True)
                #var = imgs.var(dim=-1, keepdim=True)
                imgs = (imgs - mean) / (var + 1.e-6)**.5
                imgs = self.unpatchify(imgs)
        else:
            imgs = self.patchify(imgs)
            if self.norm_pix_loss:
                #mean = imgs.mean(dim=-1, keepdim=True)
                #var = imgs.var(dim=-1, keepdim=True)
                # Compute mean and variance of patches in target
                mean, var = patch_mean_and_var(imgs)
                imgs = (imgs - mean) / (var + 1.e-6)**.5
    
        if self.loss_fn=='mse':
            loss = torch.nn.functional.mse_loss(imgs, pred, reduction='none')
        else:
            loss = torch.nn.functional.l1_loss(imgs, pred, reduction='none')

        # Replace NaN values in loss with 0
        loss = torch.nan_to_num(loss, nan=0.0)
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)
            
        return loss

    def norm_inputs(self, x):
        return (x - self.pixel_mean) / self.pixel_std
    
    def denorm_imgs(self, orig_imgs, norm_imgs):
        if self.norm_pix_loss:
            # Undo pixel norm
            norm_imgs = undo_pixel_norm(orig_imgs, norm_imgs, self)
        return norm_imgs

    def forward(self, imgs, mask_ratio=0.75, mask=None, denorm_out=False):
        imgs = self.norm_inputs(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, mask)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs.detach(), pred, mask)
        return loss, pred, mask

def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge(**kwargs):
    model = MaskedAutoencoderViT(
        depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def simmim_vit(**kwargs):
    #drop_rate = 0.0
    #drop_path_rate = 0.1
    #init_values = 0.1

    model = MaskedAutoencoderViT(
        depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mim_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mim_vit_huge(**kwargs):
    model = MaskedAutoencoderViT(
        depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def patch_mean_and_var(imgs):
    # Create a mask of non-NaN values (True where the data is not NaN)
    non_nan_mask = ~torch.isnan(imgs)
    
    # Calculate the mean of non-NaN values
    mean = torch.where(non_nan_mask, imgs, torch.tensor(0.0, device=imgs.device)).sum(dim=-1, keepdim=True) / non_nan_mask.sum(dim=-1, keepdim=True)
    
    # Calculate the variance of non-NaN values
    # Subtract the mean from only the non-NaN values and square the result.
    diff_squared = torch.where(non_nan_mask, imgs - mean, torch.tensor(0.0, device=imgs.device)) ** 2
    # Sum the squared differences, divide by the count of non-NaN values to get the variance.
    var = diff_squared.sum(dim=-1, keepdim=True) / non_nan_mask.sum(dim=-1, keepdim=True)

    return mean, var

def undo_pixel_norm(original_images, normalized_images, model):
    """
    Undo the normalization used in the pixel norm loss.

    Args:
    normalized_images (torch.Tensor): The normalized images.

    Returns:
    torch.Tensor: The unnormalized images.
    """
    
    original_images = model.patchify(original_images)
    normalized_images = model.patchify(normalized_images)
    
    mean = original_images.mean(dim=-1, keepdim=True)
    var = original_images.var(dim=-1, keepdim=True)

    unnormalized = normalized_images * (var + 1.e-6)**.5 + mean
    
    return model.unpatchify(unnormalized)