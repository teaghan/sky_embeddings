import torch

def run_iter(model, samples, masks, ra_decs, labels, optimizer, lr_scheduler,
             losses_cp, loss_fn='mse', label_uncertainties=None, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)
        
    # Run forward prop
    model_output = model(samples, mask=masks, ra_dec=ra_decs)

    # Compute loss
    if 'crossentropy' in loss_fn.lower():
        loss = torch.nn.CrossEntropyLoss()(model_output, labels.squeeze(1))
        metric = (torch.max(model_output, 1)[1] == labels.squeeze(1)).float().mean()
    if 'mse' in loss_fn.lower():
        labels = model.module.normalize_labels(labels)
        if label_uncertainties is None:
            loss = torch.nn.MSELoss()(model_output, labels)
        else:
            # Inverse uncertainties used as weights
            weights = 1.0 / (label_uncertainties + 1e-5)
            loss = torch.nn.functional.mse_loss(model_output, labels, reduction='none')
            weighted_loss = loss * weights
            loss = weighted_loss.mean()
        metric = torch.nn.L1Loss()(model_output, labels)
    
    if loss.numel()>1:
        # In case of multiple GPUs
        loss = loss.unsqueeze(0).mean()
        metric = metric.unsqueeze(0).mean()
    
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
        if 'crossentropy' in loss_fn.lower():
            losses_cp['train_acc'].append(float(metric))
        else: 
            losses_cp['train_mae'].append(float(metric))

    else:
        # Save loss and metrics
        losses_cp['val_loss'].append(float(loss))
        if 'crossentropy' in loss_fn.lower():
            losses_cp['val_acc'].append(float(metric))
        else: 
            losses_cp['val_mae'].append(float(metric))
                
    return model, optimizer, lr_scheduler, losses_cp
