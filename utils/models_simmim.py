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
from pretrain import str2bool

def build_model(config, model_filename, device, build_optimizer=False):

    # Model architecture
    norm_pix_loss = str2bool(config['TRAINING']['norm_pix_loss'])
    img_size = int(config['ARCHITECTURE']['img_size'])
    num_channels = int(config['ARCHITECTURE']['num_channels'])
    embed_dim = int(config['ARCHITECTURE']['embed_dim'])
    patch_size = int(config['ARCHITECTURE']['patch_size'])
    model_type = config['ARCHITECTURE']['model_type']
    input_norm = config['ARCHITECTURE']['input_norm']

    # Construct the model
    if model_type=='base':
        model = mae_vit_base(embed_dim=embed_dim, 
                             img_size=img_size,
                             in_chans=num_channels,
                             patch_size=patch_size,
                             norm_pix_loss=norm_pix_loss,
                             input_norm=input_norm)
    elif model_type=='large':
        model = mae_vit_large(embed_dim=embed_dim,
                              img_size=img_size,
                             in_chans=num_channels,
                             patch_size=patch_size,
                             norm_pix_loss=norm_pix_loss,
                             input_norm=input_norm)
    elif model_type=='huge':
        model = mae_vit_huge(embed_dim=embed_dim,
                             img_size=img_size,
                             in_chans=num_channels,
                             patch_size=patch_size,
                             norm_pix_loss=norm_pix_loss,
                             input_norm=input_norm)
    elif model_type=='simmim':
        model = simmim_vit(embed_dim=embed_dim,
                           img_size=img_size,
                           in_chans=num_channels,
                           patch_size=patch_size,
                           norm_pix_loss=norm_pix_loss,
                           input_norm=input_norm,
                           simmim=True)
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
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, input_norm=None, simmim=False):
        super().__init__()

        self.simmim = simmim
        
        if 'layer' in input_norm.lower():
            self.input_norm = nn.LayerNorm([in_chans, img_size, img_size], elementwise_affine=True)
        elif 'batch' in input_norm.lower():
            self.input_norm = nn.BatchNorm2d(in_chans)
        elif 'group' in input_norm.lower():
            self.input_norm = nn.GroupNorm(1, in_chans)
        else:
            self.input_norm = None
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
            # Mask 
            self.mask_token = nn.Parameter(torch.zeros((in_chans, patch_size, patch_size)))

            # Simple decoder
            encoder_stride = img_size//patch_size
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=embed_dim,
                    out_channels=encoder_stride ** 2 * in_chans, kernel_size=1),
                nn.PixelShuffle(encoder_stride),
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
    
                # Mask values are the same for every patch. Need to repeat these to create 
                # an array that is the same size as the image
                tile_size = (H // self.mask_token.shape[1], W // self.mask_token.shape[2])
                patch_mask_values = self.mask_token.repeat(1, tile_size[0], tile_size[1])
                
                # Image is masked where mask==1 and replaced with the values in patch_mask_values
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
        if self.simmim:
            if self.norm_pix_loss:
                imgs = self.patchify(imgs)
                mean = imgs.mean(dim=-1, keepdim=True)
                var = imgs.var(dim=-1, keepdim=True)
                imgs = (imgs - mean) / (var + 1.e-6)**.5
                imgs = self.unpatchify(imgs)
            
            loss_recon = torch.nn.functional.l1_loss(imgs, pred, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
            
        else:
            target = self.patchify(imgs)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
    
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            
        return loss

    def denorm_imgs(self, orig_imgs, norm_imgs):
        if self.norm_pix_loss:
            if self.input_norm:
                # First apply input norm to get correct normed statistics
                orig_imgs_normed = self.input_norm(orig_imgs)
            else:
                orig_imgs_normed = orig_imgs
            # Undo pixel norm
            norm_imgs = undo_pixel_norm(orig_imgs_normed, norm_imgs, self)
            
        # Now undo input norm with orig images
        if type(self.input_norm)==nn.LayerNorm:
            return undo_layer_norm(orig_imgs, norm_imgs, self.input_norm)
        elif type(self.input_norm)==nn.GroupNorm:
            return undo_group_norm(orig_imgs, norm_imgs, self.input_norm)
        elif type(self.input_norm)==nn.BatchNorm2d:
            return undo_batch_norm(orig_imgs, norm_imgs, self.input_norm)
        else:
            return norm_imgs
        

    def forward(self, imgs, mask_ratio=0.75, mask=None, denorm_out=False):
        if self.input_norm:
            imgs = self.input_norm(imgs)
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

def undo_layer_norm(original_images, normalized_images, layer_norm):
    """
    Undo the normalization by LayerNorm, including the epsilon value.

    Args:
    normalized_images (torch.Tensor): The normalized images.
    layer_norm (torch.nn.LayerNorm): The LayerNorm layer used for normalization.

    Returns:
    torch.Tensor: The unnormalized images.
    """
    
    original_means = torch.mean(original_images, dim=(1,2,3), keepdim=True)
    original_vars = torch.var(original_images, dim=(1,2,3), keepdim=True, unbiased=False)
    
    # Get the epsilon value used in LayerNorm
    epsilon = layer_norm.eps

    if layer_norm.elementwise_affine:
        # Ensure that bias and weight are not None
        bias = layer_norm.bias if layer_norm.bias is not None else 0
        weight = layer_norm.weight if layer_norm.weight is not None else 1

        # Reverse the affine transformation
        unnormalized = (normalized_images - bias) / weight
    else:
        unnormalized = normalized_images

    # Reverse the standard normalization, including epsilon for stability
    unnormalized = unnormalized * torch.sqrt(original_vars + epsilon) + original_means

    return unnormalized

def undo_group_norm(original_images, normalized_images, group_norm):
    """
    Undo the normalization by GroupNorm, including the epsilon value.

    Args:
    normalized_images (torch.Tensor): The normalized images.
    group_norm (torch.nn.GroupNorm): The GroupNorm layer used for normalization.

    Returns:
    torch.Tensor: The unnormalized images.
    """
    N, C, H, W = original_images.size()
    num_groups = group_norm.num_groups
    # Compute original means and variances for each group
    group_size = C // num_groups
    original_means = original_images.view(N, num_groups, group_size, H, W).mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
    original_vars = original_images.view(N, num_groups, group_size, H, W).var(dim=(2, 3, 4), keepdim=True, unbiased=False).squeeze(2)
    
    # Get the epsilon value used in GroupNorm
    epsilon = group_norm.eps

    if group_norm.affine:
        # Ensure that bias and weight are not None
        bias = group_norm.bias.view(1,-1,1,1) if group_norm.bias is not None else 0
        weight = group_norm.weight.view(1,-1,1,1) if group_norm.weight is not None else 1            
        
        # Reverse the affine transformation
        unnormalized = (normalized_images - bias) / weight
    else:
        unnormalized = normalized_images

    # Reverse the standard normalization, including epsilon for stability
    unnormalized = unnormalized * torch.sqrt(original_vars + epsilon) + original_means

    return unnormalized

def undo_batch_norm(original_images, normalized_images, batch_norm):
    """
    Attempt to undo the normalization by BatchNorm2d.

    Args:
    normalized_images (torch.Tensor): The normalized images.
    batch_norm (torch.nn.BatchNorm2d): The BatchNorm2d layer used for normalization.

    Returns:
    torch.Tensor: The unnormalized images.
    """
    # Compute original means and variances per channel
    original_means = original_images.mean(dim=(0, 2, 3))
    original_vars = original_images.var(dim=(0, 2, 3), unbiased=False)
    
    if batch_norm.affine:
        # Ensure that bias and weight are not None
        bias = batch_norm.bias if batch_norm.bias is not None else 0
        weight = batch_norm.weight if batch_norm.weight is not None else 1

        # Reverse the affine transformation
        unnormalized = (normalized_images - bias[None, :, None, None]) / weight[None, :, None, None]
    else:
        unnormalized = normalized_images

    # Reverse the standard normalization
    unnormalized = unnormalized * torch.sqrt(original_vars[None, :, None, None] + batch_norm.eps) + original_means[None, :, None, None]

    return unnormalized