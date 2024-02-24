# Masked Image Modelling Configuration Parameters

These parameters control various aspects of data handling, training, and model architecture for the masked image modelling.

## [DATA]
- `train_data_paths`: A list of directories containing the training FITS files. Example: `['/home/user/scratch/sky_embeddings/data/pdr3_wide', '/home/user/scratch/sky_embeddings/data/pdr3_dud']`.
- `bands`: Lists the color bands (e.g., ['G','I','R','Y','Z']) used in the analysis. These bands represent different filters used during the observation.
- `min_bands`: The minimum number of bands required to include a given patch of sky in the dataset. Set to 5 by default. If `min_bands` is less than the number of `bands`, then any missing bands will be replaced by a learnable set of parameters within the model.
- `cutouts_per_tile`: Number of random cutouts to create per FITS tile, set to 2048 by default. This means that - during training - each worker will load a separate tile in the sky and create this many cutouts per tile. These cutouts will then be split into batches to iterate over during training.
- `val_data_file`: Filename for the validation dataset, such as `HSC_galaxies_GRIZY_64_val_new.h5`. At a minimum, the dataset should include the elements `cutouts`, `ra`, and `dec`.
- `pos_channel`: Indicates whether or not to use the positional channel. This is set as a boolean (`False` by default). **Note:** this capability is currently in development and shouldn't be used yet.
- `lp_class_data_file`: Filename for the linear probing classification validation dataset, such as `simple_classifier_data.h5`. At a minimum, the dataset should include the elements `cutouts`, `class` (which includes integer values starting at `0` and going to `num_classes-1` denoting the class of each object). If you do not want to use this validation process, simply remove this parameter from your config file.
- `lp_regress_data_file`: Filename for the linear probing classification validation dataset, such as `simple_classifier_data.h5`. At a minimum, the dataset should include the elements `cutouts`, `zspec`. If you do not want to use this validation process, simply remove this parameter from your config file.

## [TRAINING]
- `batch_size`: The size of the batch used during training, defaulting to 64.
- `total_batch_iters`: Total number of batch iterations for training, set to 1,000,000 by default.
- `max_mask_ratio`: During training, each sample will have a different fraction of its patches masked. This parameter sets the maximum fraction of patches that will be masked during training, set to 0.9 by default. 
- `norm_pix_loss`: Indicates whether to normalize pixel loss. This is a boolean value (`True` by default).
- `weight_decay`: Weight decay parameter for the optimizer, set to 0.05 by default.
- `init_lr`: Initial learning rate for training, set to 0.0001 by default.
- `final_lr_factor`: Factor by which the final learning rate is reduced, set to 10,000,000 by default.
- `loss_fn`: Specifies the loss function to use, either `MSE` or `L1`. The default is `L1`.

## [ARCHITECTURE]
- `img_size`: Dimensions of the images (number of rows and columns in each image sample), set to 64 by default.
- `num_channels`: Number of channels in each image sample, corresponding to the number of color bands used, set to 5 by default.
- `pixel_mean`: Mean pixel value used to normalize model inputs, set to 0.0 by default.
- `pixel_std`: Standard deviation of pixel values used to normalize model inputs, set to 1.0 by default.
- `embed_dim`: Size of the embeddings in the encoder, set to 768 by default.
- `patch_size`: Dimensions of each patch from the image samples, set to 8 by default.
- `model_type`: Specifies the model type or size, indicating the architecture template to use. The default choice is `simmim` which is the Simple Masked Image Modelling architecure. Other options currently include `mimlarge` and `mimhuge`. These options will likely change over time as further testing is done.

## [Notes]
- `comment`: A field for any additional comments or notes regarding the configuration. This is useful to keep track of the changes from previous iterations of training.

