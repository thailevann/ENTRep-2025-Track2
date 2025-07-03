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

def compute_classification_accuracy(model, triplets, cls_map, class2idx, device, steps=10):
    """
    Compute classification accuracy for triplet learning
    
    Args:
        model: Vector field model
        triplets: List of triplets
        cls_map: Class mapping dictionary
        class2idx: Class to index mapping
        device: Device to run on
        steps: Integration steps
        
    Returns:
        float: Classification accuracy (0.0 to 1.0)
    """
    model.eval()
    total_correct = 0
    total_samples = 0

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

            # Apply vector field transformation
            pred_embeddings = euler_integration(batch_embeddings, model, steps=steps)
            
            # For triplet learning, we can use similarity-based classification
            # Group by anchor, positive, negative triplets
            batch_size = len(batch)
            anchors = pred_embeddings[::3]  # Every 3rd starting from 0
            positives = pred_embeddings[1::3]  # Every 3rd starting from 1
            negatives = pred_embeddings[2::3]  # Every 3rd starting from 2
            
            # Compute similarities
            anchor_pos_sim = F.cosine_similarity(anchors, positives, dim=1)
            anchor_neg_sim = F.cosine_similarity(anchors, negatives, dim=1)
            
            # Count correct triplets (anchor closer to positive than negative)
            correct_triplets = (anchor_pos_sim > anchor_neg_sim).sum().item()
            total_correct += correct_triplets
            total_samples += batch_size

    return total_correct / total_samples if total_samples > 0 else 0.0

def compute_triplet_accuracy(model, triplets, cls_map, class2idx, device, steps=10):
    """
    Compute triplet accuracy - percentage of correctly ordered triplets
    
    Args:
        model: Vector field model
        triplets: List of triplets
        cls_map: Class mapping dictionary
        class2idx: Class to index mapping
        device: Device to run on
        steps: Integration steps
        
    Returns:
        float: Triplet accuracy (0.0 to 1.0)
    """
    return compute_classification_accuracy(model, triplets, cls_map, class2idx, device, steps)

def compute_embedding_classification_accuracy(embeddings, labels, num_classes):
    """
    Compute classification accuracy using nearest centroid classification
    
    Args:
        embeddings: Tensor of embeddings [N, D]
        labels: Tensor of true labels [N]
        num_classes: Number of classes
        
    Returns:
        float: Classification accuracy
    """
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute class centroids
    centroids = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            class_embeddings = embeddings[class_mask]
            centroid = class_embeddings.mean(dim=0)
            centroids.append(F.normalize(centroid.unsqueeze(0), dim=1))
        else:
            # Random centroid if no samples for this class
            centroids.append(F.normalize(torch.randn(1, embeddings.size(1)), dim=1))
    
    centroids = torch.cat(centroids, dim=0)  # [num_classes, D]
    
    # Compute similarities to centroids
    similarities = torch.mm(embeddings, centroids.t())  # [N, num_classes]
    predictions = torch.argmax(similarities, dim=1)
    
    accuracy = (predictions == labels).float().mean().item()
    return accuracy

def log_training_progress(epoch, train_loss, test_loss, lr, train_acc=None, test_acc=None, log_interval=10):
    """
    Log training progress including accuracy/recall metrics
    
    Args:
        epoch: Current epoch
        train_loss: Training loss
        test_loss: Test loss
        lr: Current learning rate
        train_acc: Training accuracy/recall (optional)
        test_acc: Test accuracy/recall (optional)
        log_interval: How often to log
    """
    if (epoch + 1) % log_interval == 0:
        log_msg = f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}"
        
        if train_acc is not None:
            log_msg += f", Train Recall@1 = {train_acc:.4f}"
        if test_acc is not None:
            log_msg += f", Test Recall@1 = {test_acc:.4f}"
            
        log_msg += f", LR = {lr:.1e}"
        print(log_msg)

# Example usage
if __name__ == "__main__":
    print("Testing training utilities...")
    
    # Test basic functionality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Training utilities ready for use!")