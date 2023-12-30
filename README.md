# Sky Embeddings

Using self-supervised learning to create embeddings of images of the sky.

These embeddings can then be used for down-stream tasks such as classification, estimating redshifts, and performing similarity searches.

## Masked Autoencoders

So far, we are using the [MAE developed by facebook](https://github.com/facebookresearch/mae) as our ML framework.

## Dependencies

- Python 3.11.5

- [PyTorch](http://pytorch.org/): `pip install torch==2.0.1`

- h5py: `pip install h5py`

## Hyper Suprime-Cam (HSC) - Subaru Telescope

The work in this repo shows an application to images taken by the HSC on the Subaru Telescope.

<p align="left">
  <img width="1000" height="1000" src="./figures/hsc_subaru.jpg">
</p>

## Data download

(still to come)

## Training the Network

### Option 1

1. The model architecture and training parameters are set within configuration file in [the config directory](./configs). For instance, I have already created the [original configuration file](./configs/mae_1.ini). You can copy this file under a new name and change whichever parameters you choose.
  
2. If you were to create a config file called `mae_2.ini` in Step 1, this model could be trained by running `python train_mae.py mae_2 -v 5000 -ct 10.00` which will train your model displaying the progress every 5000 batch iterations and the model would be saved every 10 minutes. This same command will continue training the network if you already have the model saved in the [model directory](./models) from previous training iterations. 

### Option 2

Alternatively, if operating on compute-canada, you can use the `cc/launch_mae.py` script to simultaneously create a new configuration file and launch a bunch of jobs to train your model. 

1. Change the [load modules file](./cc/module_loads.txt) to include the lines necessary to load your own environment with pytorch, etc. 
2. Then, to copy the [original configuration](./configs/mae_1.ini), but use, say, a batch size of 32 images, you could use the command `python cc/launch_mae.py mae_2 -bs 32`. This will launch five 3-hour jobs on the GPU nodes to complete the training. You can checkout the other parameters that can be changed with the command `python cc/launch_mae.py -h`.

## Analysis notebooks

1. Checkout the [test notebook](./test_mae.ipynb) to evaluate the trained MAE.
2. Checkout the [similarity search notebook](./latent_similarity.ipynb) for our developing work on using the embeddings to do similarity searches against images of known object classes.
