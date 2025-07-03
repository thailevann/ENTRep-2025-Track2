"""
Training-specific utilities for ENTRep 2025 Track 2
Functions specifically for training workflows
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

def euler_integration(x0, vf, steps=10):
    """
    Euler integration with RK4 method for Vector Field
    
    Args:
        x0: Initial embeddings [B, D]
        vf: Vector field model (takes in x and t, returns dx/dt)
        steps: Number of integration steps
        
    Returns:
        torch.Tensor: Transformed embeddings x(T)
    """
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = i * dt
        k1 = vf(x, t)
        k2 = vf(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vf(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = vf(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def compute_triplet_loss(model, triplets, loss_func, cls_map, class2idx, device, steps=10):
    """
    Compute loss using MultiSimilarityLoss
    
    Args:
        model: Vector field model
        triplets: List of triplets
        loss_func: Loss function
        cls_map: Class mapping dictionary
        class2idx: Class to index mapping
        device: Device to run on
        steps: Integration steps
        
    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(triplets), 64):
            batch = triplets[i : i + 64]
            if len(batch) == 0:
                continue

            batch_embeddings, batch_labels = get_embeddings_labels_from_triplets(batch, cls_map, class2idx)
            if batch_embeddings is None or len(batch_embeddings) == 0:
                continue

            batch_embeddings = batch_embeddings.to(device).float()
            batch_labels = batch_labels.to(device).long()

            pred_embeddings = euler_integration(batch_embeddings, model, steps=steps)
            loss = loss_func(pred_embeddings, batch_labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)

def get_embeddings_labels_from_triplets(triplets_batch, cls_map, class2idx):
    """
    Convert list of triplets into tensors for model input
    
    Args:
        triplets_batch: List of triplets
        cls_map: Class mapping dictionary
        class2idx: Class to index mapping
        
    Returns:
        tuple: (embeddings_tensor, labels_tensor)
    """
    embeddings = []
    labels = []

    for anchor_emb, positive_emb, negative_emb, anchor_name, negative_name in triplets_batch:
        anchor_cls = cls_map[anchor_name]
        negative_cls = cls_map[negative_name]

        embeddings.extend([anchor_emb, positive_emb, negative_emb])
        labels.extend([
            class2idx[anchor_cls],
            class2idx[anchor_cls],  # Positive has same class as anchor
            class2idx[negative_cls]
        ])

    if not embeddings:
        return None, None
        
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels)
    
    return embeddings_tensor, labels_tensor

def setup_optimizer_and_scheduler(model, lr=1e-4, scheduler_patience=15, scheduler_factor=0.8):
    """
    Setup optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        lr: Learning rate
        scheduler_patience: Patience for LR scheduler
        scheduler_factor: Factor to reduce LR
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    
    return optimizer, scheduler

def update_learning_rate(optimizer, scheduler, loss, epoch, warmup_epochs=20):
    """
    Update learning rate with warmup and scheduling
    
    Args:
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
        loss: Current loss for scheduling
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        
    Returns:
        bool: True if LR was reduced, False otherwise
    """
    old_lr = optimizer.param_groups[0]['lr']
    
    if epoch >= warmup_epochs:
        scheduler.step(loss)
    else:
        scheduler.step(float('inf'))  # Don't reduce LR during warmup

    new_lr = optimizer.param_groups[0]['lr']
    
    if new_lr < old_lr:
        print(f"LR reduced at epoch {epoch + 1} → {new_lr:.1e}")
        return True
    
    return False

def check_early_stopping(current_loss, best_loss, epochs_no_improve, patience=40, delta=5e-4):
    """
    Check if training should stop early
    
    Args:
        current_loss: Current epoch loss
        best_loss: Best loss seen so far
        epochs_no_improve: Number of epochs without improvement
        patience: Patience for early stopping
        delta: Minimum change to count as improvement
        
    Returns:
        tuple: (should_stop, new_best_loss, new_epochs_no_improve)
    """
    if current_loss < best_loss - delta:
        # Significant improvement
        return False, current_loss, 0
    else:
        # No improvement
        epochs_no_improve += 1
        should_stop = epochs_no_improve >= patience
        return should_stop, best_loss, epochs_no_improve

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath, device):
    """
    Load training checkpoint
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load on
        
    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss

def log_training_progress(epoch, train_loss, test_loss, lr, log_interval=10):
    """
    Log training progress
    
    Args:
        epoch: Current epoch
        train_loss: Training loss
        test_loss: Test loss
        lr: Current learning rate
        log_interval: How often to log
    """
    if (epoch + 1) % log_interval == 0:
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
              f"Test Loss = {test_loss:.4f}, LR = {lr:.1e}")

# Example usage
if __name__ == "__main__":
    print("Testing training utilities...")
    
    # Test basic functionality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Training utilities ready for use!")