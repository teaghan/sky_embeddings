import logging
import os

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.dataloaders import build_h5_dataloader
from utils.distributed import AllGather, AllReduce
from utils.eval_fns import mae_latent
from utils.jepa_masking import apply_masks
from utils.jepa_tensors import repeat_interleave_batch

# from lark import logger
from utils.misc import log_memory_usage, select_centre
from utils.plotting_fns import plot_confusion_matrix

logger = logging.getLogger()


def run_iter(model, samples, ra_decs, masks, mask_ratio, optimizer, lr_scheduler, losses_cp, mode='train'):
    if mode == 'train':
        model.train(True)
    else:
        model.train(False)

    # Run predictions and calculate loss
    loss, _, _ = model(samples, ra_dec=ra_decs, mask_ratio=mask_ratio, mask=masks)
    if loss.numel() > 1:
        # In case of multiple GPUs
        loss = loss.unsqueeze(0).mean()

    if 'train' in mode:
        # Update the gradients
        loss.backward()
        # Adjust network weights
        optimizer.step()
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Adjust learning rate
        lr_scheduler.step()

        # Save loss and metrics
        losses_cp['train_loss'].append(float(loss))

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(loss))

    return model, optimizer, lr_scheduler, losses_cp


def val_iter(encoder, predictor, target_encoder, images, masks_enc, masks_pred, use_bfloat16):
    def forward_target():
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature dimension
        B = len(h)
        # create target representations
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))  # type: ignore
        return h

    def forward_context():
        z = encoder(images, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        return z

    def compute_loss(z, h):
        loss = F.smooth_l1_loss(z, h)
        loss = AllReduce.apply(loss)
        return loss

    # Forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
        h = forward_target()
        z = forward_context()
        loss = compute_loss(z, h)

    return loss


def linear_probe(
    model,
    losses_cp,
    device,
    dataloader_template,
    model_type,
    class_data_path=None,
    regress_data_path=None,
    combine='central',
    remove_cls=True,
    world_size=1,
    rank=0,
):
    """Train a quick linear probing model to evaluate the quality of the embeddings."""

    if combine == 'token':
        remove_cls = False

    model.train(False)
    if class_data_path:
        # Classifier task
        x, y = get_embeddings(
            class_data_path,
            model,
            device,
            dataloader_template,
            model_type,
            y_label='class',
            combine=combine,
            remove_cls=remove_cls,
            world_size=world_size,
            rank=rank,
        )

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Creating and training a classifier
        clf = LogisticRegression(
            solver='lbfgs', multi_class='multinomial', max_iter=10000, C=0.01, random_state=42
        )
        clf.fit(X_train, y_train)

        # Predicting the class label
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        # Evaluating the classifier
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')

        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_precision = precision_score(y_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_train, y_pred_train, average='weighted')
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')

        # Update losses_cp
        losses_cp['train_lp_acc'].append(float(train_accuracy))
        losses_cp['train_lp_precision'].append(float(train_precision))
        losses_cp['train_lp_recall'].append(float(train_recall))
        losses_cp['train_lp_f1'].append(float(train_f1))

        losses_cp['val_lp_acc'].append(float(test_accuracy))
        losses_cp['val_lp_precision'].append(float(test_precision))
        losses_cp['val_lp_recall'].append(float(test_recall))
        losses_cp['val_lp_f1'].append(float(test_f1))

    if regress_data_path:
        # Regression task
        x, y = get_embeddings(
            regress_data_path,
            model,
            device,
            dataloader_template,
            model_type,
            y_label='zspec',
            combine=combine,
            remove_cls=remove_cls,
            world_size=world_size,
            rank=rank,
        )

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Creating and training a linear model for regression
        # regressor = LinearRegression()
        regressor = ElasticNet(alpha=0.0001, l1_ratio=0.9, max_iter=10000, random_state=42)
        regressor.fit(X_train, y_train)

        # Predicting the continuous values
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)

        # Evaluating the regressor
        # mse_test = mean_squared_error(y_test, y_pred_test)
        # mse_train = mean_squared_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        losses_cp['train_lp_r2'].append(float(r2_train))
        losses_cp['val_lp_r2'].append(float(r2_test))


def get_embeddings(
    data_path,
    model,
    device,
    dataloader_template,
    model_type,
    y_label='class',
    combine='central',
    remove_cls=True,
    world_size=1,
    rank=0,
):
    # Data loader
    dataloader = build_h5_dataloader(
        data_path,
        batch_size=64,
        bands=dataloader_template.dataset.bands,
        num_workers=dataloader_template.num_workers,
        img_size=dataloader_template.dataset.img_size,
        num_patches=dataloader_template.dataset.num_patches,
        patch_size=model.module.patch_embed.patch_size
        if hasattr(model, 'module')
        else model.patch_embed.patch_size,
        num_channels=model.module.in_chans if hasattr(model, 'module') else model.in_chans,
        max_mask_ratio=None,
        shuffle=False,
        world_size=world_size,
        rank=rank,
    )

    # Map target samples to latent-space
    latent_features = mae_latent(model, dataloader, device, model_type, verbose=0, remove_cls=remove_cls)
    latent_features = latent_features.data.cpu().numpy()  # type: ignore

    # Collect targets
    with h5py.File(data_path, 'r') as f:
        y = f[y_label][:]  # type: ignore
    if 'jepa' not in model_type:
        if model.module.attn_pool:
            # There is only one output set of features if there is an attention pooling layer
            combine = 'flatten'

    scale = True
    if combine == 'token':
        x = latent_features[:, :1].reshape(latent_features.shape[0], -1)
    elif combine == 'flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    elif combine == 'pool':
        x = np.max(latent_features, axis=1)
    elif combine == 'centralpool':
        x = select_centre(latent_features, n_patches=16)
        x = np.max(x, axis=1)
    elif combine == 'central':
        x = select_centre(latent_features, n_patches=4)
        x = x.reshape(x.shape[0], -1)
    elif combine == 'mean':
        x = np.mean(latent_features, axis=1)
    else:
        x = latent_features
        x = (x - np.nanmean(x)) / np.nanstd(x)
        scale = False

    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    return x, y


def distributed_linear_probe(
    model,
    patch_size,
    num_channels,
    losses_cp,
    device,
    dataloader_template,
    model_type,
    model_name,
    class_data_path=None,
    regress_data_path=None,
    combine='central',
    remove_cls=True,
    world_size=1,
    rank=0,
    plot_conf_matrix=False,
    fig_dir='None',
):
    model.eval()

    def process_data(data_path, y_label):
        dataloader = build_h5_dataloader(
            data_path,
            batch_size=64,
            bands=dataloader_template.dataset.bands,
            num_workers=dataloader_template.num_workers,
            img_size=dataloader_template.dataset.img_size,
            patch_size=patch_size,
            num_channels=num_channels,
            max_mask_ratio=None,
            shuffle=False,
            model_type=model_type,
            world_size=world_size,
            rank=rank,
        )
        try:
            # Generate embeddings (this is distributed across all ranks)
            try:
                latent_features = mae_latent(
                    model,
                    dataloader,
                    device=device,
                    model_type=model_type,
                    verbose=1,
                    remove_cls=remove_cls,
                    world_size=world_size,
                    rank=rank,
                )
            except Exception as e:
                logger.error(f'Error in generating embeddings: {e}')
            logger.info(f'len(latent_features): {len(latent_features)}')
            if isinstance(latent_features, tuple):
                latent_features = latent_features[0]
                if len(latent_features == 2):
                    indices = latent_features[1]
                elif len(latent_features == 3):
                    indices = latent_features[1]
                    images = latent_features[2]  # noqa: F841
                else:
                    raise ValueError('Invalid number of outputs from mae_latent')

        except Exception as e:
            logger.error(f'Error in generating embeddings: {e}')

        logger.info(f'Len indices: {len(indices)}, type: {type(indices)}')
        logger.info(f'First 10 indices: {indices[:10]}')
        logger.info(f'Shape indices: {indices.shape}')

        log_memory_usage()

        # Gather embeddings from all ranks
        try:
            gathered_features = AllGather.apply(latent_features)
            log_memory_usage()
            # gathered_indices = AllGather.apply(indices)
        except Exception as e:
            logger.error(f'Error in all gather : {e}.')

        log_memory_usage()

        if rank == 0:
            # Combine gathered features
            if dist.is_available() and dist.is_initialized():
                # latent_features = torch.cat(gathered_features, dim=0)
                logger.info(f'latent_features shape: {gathered_features.shape} in distributed case.')
            else:
                logger.info(f'latent_features shape: {gathered_features.shape} in non-distributed case.')  # type: ignore

            # Reorder the features back to the original order
            # sorted_order = torch.argsort(gathered_indices).cpu().numpy()  # type: ignore
            latent_features = gathered_features.cpu().numpy()  # type: ignore
            # latent_features = latent_features[sorted_order]

            # Load labels (assuming they're stored in the same order as the data)
            with h5py.File(data_path, 'r') as f:
                y = f[y_label][:]
                assert len(y) == len(
                    latent_features
                ), f'Mismatch: {len(y)} labels, {len(latent_features)} features'

            # Process features
            try:
                x = process_features(model, model_type, latent_features, combine)
            except Exception as e:
                logger.error(f'Error in processing features: {e}')

            return x, y
        else:
            return None, None

    if class_data_path:
        x, y = process_data(class_data_path, 'class')

        if rank == 0:
            logger.info(f'Performing classification task on rank {rank}...')
            # Perform classification on rank 0
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(
                solver='lbfgs', multi_class='multinomial', max_iter=10000, C=0.01, random_state=42
            )
            clf.fit(X_train, y_train)

            # Compute metrics
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)

            # Evaluating the classifier
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')

            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train, average='weighted')
            train_recall = recall_score(y_train, y_pred_train, average='weighted')
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')

            logger.info(f'Training Accuracy: {train_accuracy:.3f}, Validation Accuracy: {test_accuracy:.3f}')
            logger.info(
                f'Training Precision: {train_precision:.3f}, Validation Precision: {test_precision:.3f}'
            )
            logger.info(f'Training Recall: {train_recall:.3f}, Validation Recall: {test_recall:.3f}')
            logger.info(f'Training F1: {train_f1:.3f}, Validation F1: {test_f1:.3f}')

            # Update losses_cp
            losses_cp['train_lp_acc'].append(float(train_accuracy))
            losses_cp['train_lp_precision'].append(float(train_precision))
            losses_cp['train_lp_recall'].append(float(train_recall))
            losses_cp['train_lp_f1'].append(float(train_f1))

            losses_cp['val_lp_acc'].append(float(test_accuracy))
            losses_cp['val_lp_precision'].append(float(test_precision))
            losses_cp['val_lp_recall'].append(float(test_recall))
            losses_cp['val_lp_f1'].append(float(test_f1))

            # Plot confusion matrix
            if plot_conf_matrix:
                plot_confusion_matrix(
                    y_test,
                    y_pred_test,
                    labels=['0, 1, 2'],
                    savename=os.path.join(fig_dir, f'{model_name}_confusion_mat.png'),
                )

    if regress_data_path:
        x, y = process_data(regress_data_path, 'zspec')

        if rank == 0:
            logger.info(f'Performing regression task on rank {rank}..')
            # Perform regression on rank 0
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            regressor = ElasticNet(alpha=0.0001, l1_ratio=0.9, max_iter=10000, random_state=42)
            regressor.fit(X_train, y_train)

            # Compute metrics
            y_pred_test = regressor.predict(X_test)
            y_pred_train = regressor.predict(X_train)
            r2_test = r2_score(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)

            logger.info(f'Training R2: {r2_train:.3f}, Validation R2: {r2_test:.3f}')

            # Update losses_cp
            losses_cp['val_lp_r2'].append(float(r2_test))
            losses_cp['train_lp_r2'].append(float(r2_train))


def process_features(model, model_type, latent_features, combine):
    if 'jepa' not in model_type:
        if hasattr(model, 'module'):
            if model.module.attn_pool:
                combine = 'flatten'
        elif model.attn_pool:
            combine = 'flatten'

    if combine == 'token':
        x = latent_features[:, :1].reshape(latent_features.shape[0], -1)
    elif combine == 'flatten':
        x = latent_features.reshape(latent_features.shape[0], -1)
    elif combine == 'pool':
        x = np.max(latent_features, axis=1)
    elif combine == 'centralpool':
        x = select_centre(latent_features, n_patches=16)
        x = np.max(x, axis=1)
    elif combine == 'central':
        x = select_centre(latent_features, n_patches=4)
        x = x.reshape(x.shape[0], -1)
    elif combine == 'mean':
        x = np.mean(latent_features, axis=1)
    else:
        x = latent_features
        x = (x - np.nanmean(x)) / np.nanstd(x)
        return x

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x


def log_current_status(
    cur_iter, total_batch_iters, losses, lp_class_data_file=None, lp_regress_data_file=None
):
    # Print current status
    logger.info(f'Batch Iterations: {cur_iter}/{total_batch_iters}')
    logger.info('Losses:')
    logger.info('Training Dataset')
    logger.info(f'  Total Loss: {losses["train_loss"][-1]:.3f}')
    logger.info('  Validation Dataset')
    logger.info(f'  Total Loss: {losses["val_loss"][-1]:.3f}')
    if lp_class_data_file is not None or lp_regress_data_file is not None:
        logger.info('Linear Probing Results:')
        if lp_class_data_file:
            logger.info('Classification Accuracy:')
            logger.info(
                f'  Training: {losses["train_lp_acc"][-1]:.3f}, Validation: {losses["val_lp_acc"][-1]:.3f}'
            )
            logger.info('Classification Precision:')
            logger.info(
                f'  Training: {losses["train_lp_precision"][-1]:.3f}, Validation: {losses["val_lp_precision"][-1]:.3f}'
            )
            logger.info('Classification Recall:')
            logger.info(
                f'  Training: {losses["train_lp_recall"][-1]:.3f}, Validation: {losses["val_lp_recall"][-1]:.3f}'
            )
            logger.info('Classification F1:')
            logger.info(
                f'  Training: {losses["train_lp_f1"][-1]:.3f}, Validation: {losses["val_lp_f1"][-1]:.3f}'
            )
        if lp_regress_data_file:
            logger.info('Regression R2')
            logger.info(
                f' Training: {losses["train_lp_r2"][-1]:.3f}, Validation: {losses["val_lp_r2"][-1]:.3f}'
            )
