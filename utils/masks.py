# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
from multiprocessing import Value
import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

class SimpleMaskCollator(object):
    def __init__(
        self,
        input_size=(32, 32),
        patch_size=8,
        nenc=1,  # Number of encoder masks per image
        npred=2,  # Number of predictor masks per image
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.nenc = nenc
        self.npred = npred
        # Shared counter for iteration tracking across processes.
        self._itr_counter = Value('i', -1)

    def step(self):
        """Increments and returns the global step counter."""
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def __call__(self, batch):
        if batch[0].ndim > 4:
            # Batch size of one from dataloader with embedded batches
            N, M, C, H, W = batch[0].size()
            collated_batch = batch[0].reshape((N*M, C, H, W))
            reshape_out = True
        else:
            collated_batch = torch.utils.data.default_collate(batch)
            reshape_out = False
        # Batch size
        B = len(collated_batch)

        n_patches = self.height * self.width
        n_per_mask = n_patches//(self.nenc + self.npred)
        
        collated_masks_enc, collated_masks_pred = [], []
        for _ in range(B):
                        
            # Generate random order of patches
            patch_indices = torch.randperm(n_patches)
            
            # Assign different patches to each mask
            masks_e = [torch.sort(patch_indices[i*n_per_mask:(i+1)*n_per_mask]).values for i in range(self.nenc)]
            start_i = self.nenc*n_per_mask
            masks_p = [torch.sort(patch_indices[start_i+i*n_per_mask:start_i+(i+1)*n_per_mask]).values for i in range(self.npred)]
            collated_masks_enc.append(masks_e)
            collated_masks_pred.append(masks_p)

        #collated_masks_pred = [[cm for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        #collated_masks_enc = [[cm for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        # Adjusting the structure for encoder and predictor masks
        if reshape_out:
            collated_batch = collated_batch.reshape((N, M, C, H, W))
            collated_masks_enc = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_enc]))
            collated_masks_pred = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_pred]))
            
        return collated_batch, collated_masks_enc, collated_masks_pred

'''
class MaskCollator(object):
    """Initializes a MaskCollator for dynamic mask generation for image batches.

    This class generates encoder and predictor masks with customizable scales, aspect ratios,
    and patch sizes for each image in a batch. It supports constraints on mask overlap and ensures
    minimum patch coverage.

    Attributes:
        input_size (tuple): The size of the input images (height, width).
        patch_size (int): The size of each patch (squared).
        enc_mask_scale (tuple): Scale range for encoder masks.
        pred_mask_scale (tuple): Scale range for predictor masks.
        aspect_ratio (tuple): Aspect ratio range for masks.
        nenc (int): Number of encoder masks to generate per image.
        npred (int): Number of predictor masks to generate per image.
        min_keep (int): Minimum number of patches to keep in a mask.
        allow_overlap (bool): Whether to allow overlap between encoder and predictor masks.
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        # Calculate the number of patches across the input dimensions.
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        # Shared counter for iteration tracking across processes.
        self._itr_counter = Value('i', -1)

    def step(self):
        """Increments and returns the global step counter."""
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        """Samples a block size based on scale and aspect ratio constraints."""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))

        h = min(h, self.height - 1)
        w = min(w, self.width - 1)

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """Generates a mask and its complement for a given block size."""
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Restricts the mask to acceptable regions, adjusting for retries."""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = original_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1

            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep

            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = original_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top + h, left:left + w] = 0

        return mask.squeeze(), mask_complement

    def __call__(self, batch):
        """Generates encoder and predictor masks for the images in a batch."""
        if batch[0].ndim>4:
            # Batch size of one from dataloader with embedded batches
            N, M, C, H, W = batch[0].size()
            collated_batch = batch[0].reshape((N*M, C, H, W))
            reshape_out = True
        else:
            collated_batch = torch.utils.data.default_collate(batch)
            reshape_out = False
        B = len(collated_batch)
        
        # Use a consistent seed for each call to ensure reproducibility.
        seed = self.step()
        g = torch.Generator().manual_seed(seed)

        # Determine the sizes of blocks for prediction and encoding masks.
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred, min_keep_enc = self.height * self.width, self.height * self.width

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))

            collated_masks_pred.append(masks_p)
            acceptable_regions = masks_C if not self.allow_overlap else None

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))

            collated_masks_enc.append(masks_e)

        # Adjust mask lengths to the smallest size across the batch.
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)


        if reshape_out:
            collated_batch = collated_batch.reshape((N, M, C, H, W))
            collated_masks_enc = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_enc]))
            collated_masks_pred = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_pred]))

            #collated_masks_enc = [mask.reshape((N, M, -1)) for mask in collated_masks_enc]
            #collated_masks_pred = [mask.reshape((N, M, -1)) for mask in collated_masks_pred]
        
        return collated_batch, collated_masks_enc, collated_masks_pred
'''

class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            ''' Helper to restrict given mask to a set of acceptable regions '''
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        if batch[0].ndim>4:
            # Batch size of one from dataloader with embedded batches
            N, M, C, H, W = batch[0].size()
            collated_batch = batch[0].reshape((N*M, C, H, W))
            reshape_out = True
        else:
            collated_batch = torch.utils.data.default_collate(batch)
            reshape_out = False
        B = len(collated_batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        if reshape_out:
            collated_batch = collated_batch.reshape((N, M, C, H, W))
            collated_masks_enc = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_enc]))
            collated_masks_pred = list(zip(*[mask.reshape((N, M, -1)) for mask in collated_masks_pred]))

        return collated_batch, collated_masks_enc, collated_masks_pred
