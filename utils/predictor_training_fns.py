import torch

def run_iter(model, samples, masks, labels, optimizer, lr_scheduler,
             losses_cp, mode='train'):
        
    if mode=='train':
        model.train(True)
    else:
        model.train(False)
        
    # Run forward prop
    model_output = model(samples, mask=masks)

    # Compute loss
    labels = model.module.normalize_labels(labels)
    loss = torch.nn.MSELoss()(model_output, labels)

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
