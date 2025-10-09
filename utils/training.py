"""
Training utilities for video generation
"""

import torch
from pathlib import Path


def get_optimizer(model, config):
    """Get optimizer from config"""
    optimizer_type = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999))
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """Get learning rate scheduler from config"""
    scheduler_type = config.get('scheduler', 'constant')
    
    if scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'linear':
        num_training_steps = config.get('num_training_steps', 10000)
        warmup_steps = config.get('warmup_steps', 500)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save training checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return model, optimizer, epoch, loss

