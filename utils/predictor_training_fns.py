import torch

def run_iter(model, samples, masks, labels, optimizer, lr_scheduler,
             losses_cp, label_uncertainties=None, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)
        
    # Run forward prop
    model_output = model(samples, mask=masks)

    # Compute loss
    labels = model.module.normalize_labels(labels)
    if label_uncertainties is None:
        loss = torch.nn.MSELoss()(model_output, labels)
    else:
        # Inverse uncertainties used as weights
        weights = 1.0 / (label_uncertainties + 1e-8)
        loss = torch.nn.functional.mse_loss(model_output, labels, reduction='none')
        weighted_loss = loss * weights
        loss = weighted_loss.mean()
    
    if loss.numel()>1:
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
