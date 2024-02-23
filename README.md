# Sky Embeddings

Welcome to the Sky Embeddings repository, where we leverage self-supervised learning to generate and utilize embeddings from sky images for tasks such as classification, redshift estimation, and similarity searches.

## Overview

This repository hosts code and methodologies for applying Masked Autoencoders (MAEs) to astronomical images, focusing on producing high-quality embeddings that capture the rich, underlying structures of the universe.

### Related Work on Masked Image Modelling

We combined the [MAE code developed by Facebook AI](https://github.com/facebookresearch/mae) and the [SimMIM Framework for Masked Image Modeling](https://github.com/microsoft/SimMIM) as our primary machine learning framework. This allows us to create meaningful embeddings from partial observations of the sky.

### Dependencies

Ensure you have the following installed:

- Python 3.11.5
- PyTorch: `pip install torch==2.0.1`
- h5py: `pip install h5py`
- Scikit-learn `pip install scikit-learn`

## Dataset: Hyper Suprime-Cam (HSC) - Subaru Telescope

Our primary dataset comes from the Hyper Suprime-Cam (HSC) on the Subaru Telescope. Below is an example image from the HSC:

<p align="center">
  <img width="600" height="600" src="./figures/hsc_subaru.jpg"><br>
  <span style="display: block; text-align: right;"><a href="https://subarutelescope.org/en/news/topics/2017/02/27/2459.html">subarutelescope.org</a></span>
</p>

### Data Download

Details on how to access and prepare the HSC data will be provided soon.

## Training the Network

You can train the network using one of the following methods:

### Option 1: Local Training

1. Set model architecture and parameters using a configuration file in [the config directory](./configs). Duplicate the [original configuration file](./configs/mim_1.ini) and modify as needed.
2. To train a model with a new config file named `mim_2.ini`, use `python train_mae.py mim_2 -v 5000 -ct 10.00`, which will train your model displaying the progress every 5000 batch iterations and the model would be saved every 10 minutes. The script will also continue training from the last save point.

### Option 2: Compute Canada Cluster

For those with access to Compute Canada:

1. Modify the [load modules file](./cc/module_loads.txt) to load the necessary environment.
2. To launch training with a modified batch size or other parameters, use `python cc/launch_pretraining.py mim_2 -bs 32`. This script automatically creates a new configuration and initiates multiple training jobs. You can checkout the other parameters that can be changed using the command `python cc/launch_pretraining.py -h`.

## Analysis Notebooks

Analysis notebooks still to come.

### Contribution and Support

We welcome contributions and suggestions! Please raise issues or submit pull requests on GitHub for any features or problems. For support, refer to the repository's issues section or contact the maintainers directly.

---

Embark on a journey of exploring the universe with machine learning through Sky Embeddings!
