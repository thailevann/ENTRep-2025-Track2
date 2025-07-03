#!/usr/bin/env python3
"""
ENTRep 2025 Track 2 Training Script
Vector Field Model with Flow Matching for Entity Recognition
"""

import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
from pytorch_metric_learning import losses
from random import shuffle
from tqdm import tqdm
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import matplotlib.pyplot as plt

from models import VectorField, GaussianFourierProjection
from utils import (
    load_images_recursively, 
    create_cls_map, 
    generate_parent_child,
    create_triplets,
    filter_parent_child,
    split_embeddings,
    filter_cls_map,
    get_triplet_batch,
    get_embeddings_labels_from_triplets,
    print_class_distribution
)

def setup_device_and_model():
    """Setup device and load CLIP model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model (ViT-B/32)
    model, preprocess = clip.load("ViT-B/32", device=device)
    return device, model, preprocess

def setup_data(img_dir, device, model, preprocess):
    """Load and process all image data"""
    print("Loading and embedding images...")
    embeddings = load_images_recursively(img_dir, device, model, preprocess)
    print(f"Total images loaded: {len(embeddings)}")
    
    # Create class mapping
    cls_map = create_cls_map(img_dir)
    print(f"Created cls_map for {len(cls_map)} images")
    
    # Generate parent-child relationships
    parent_child, class_to_imgs = generate_parent_child(cls_map, embeddings, num_children=3)
    print(f"Generated parent-child pairs for {len(parent_child)} images")
    
    return embeddings, cls_map, parent_child, class_to_imgs

def create_train_test_splits(embeddings, cls_map, parent_child):
    """Create train/test splits and triplets"""
    print("Creating train/test splits...")
    
    # Split embeddings into train and test sets
    train_embeddings, test_embeddings = split_embeddings(embeddings, cls_map)
    
    # Filter class mappings
    train_cls_map = filter_cls_map(train_embeddings, cls_map)
    test_cls_map = filter_cls_map(test_embeddings, cls_map)
    
    # Filter parent-child structure
    parent_child_train = filter_parent_child(parent_child, train_embeddings)
    parent_child_test = filter_parent_child(parent_child, test_embeddings)
    
    # Create triplets
    train_triplets = create_triplets(train_embeddings, train_cls_map, parent_child_train)
    test_triplets = create_triplets(test_embeddings, test_cls_map, parent_child_test)
    
    # Print distributions
    print("\nTrain class distribution:")
    print_class_distribution(train_cls_map)
    print("\nTest class distribution:")
    print_class_distribution(test_cls_map)
    
    return train_triplets, test_triplets, train_cls_map, test_cls_map

def euler_integration(x0, vf, steps=10):
    """
    Euler integration with RK4 method
    Args:
        x0: initial embeddings [B, D]
        vf: vector field model (takes in x and t, returns dx/dt)
        steps: number of integration steps
    Returns:
        Transformed embeddings x(T)
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

def compute_triplet_loss(model, triplets, loss_func, test_cls_map, class2idx, device, steps=10):
    """Compute loss using MultiSimilarityLoss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(triplets), 64):
            batch = triplets[i : i + 64]
            if len(batch) == 0:
                continue

            batch_embeddings, batch_labels = get_embeddings_labels_from_triplets(batch, test_cls_map, class2idx)
            if batch_embeddings is None or len(batch_embeddings) == 0:
                continue

            batch_embeddings = batch_embeddings.to(device).float()
            batch_labels = batch_labels.to(device).long()

            pred_embeddings = euler_integration(batch_embeddings, model, steps=steps)
            loss = loss_func(pred_embeddings, batch_labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)

def train_model(train_triplets, test_triplets, train_cls_map, test_cls_map, device):
    """Main training loop"""
    # Class to index mapping
    class2idx = {
        'ear-left': 3,
        'ear-right': 2,
        'nose-left': 1,
        'nose-right': 0,
        'throat': 6,
        'vc-closed': 4,
        'vc-open': 5
    }
    
    # Initialize model, loss, optimizer, scheduler
    embed_dim = 512
    vf = VectorField(embed_dim).to(device).float()
    optimizer = torch.optim.AdamW(vf.parameters(), lr=1e-4)
    loss_func = losses.MultiSimilarityLoss()
    
    # Hyperparameters
    warmup_epochs = 20
    early_stop_patience = 40
    scheduler_patience = 15
    scheduler_factor = 0.8
    epochs = 200
    batch_size = 32
    delta = 5e-4  # min change in loss to count as improvement
    
    # LR Scheduler & Early Stopping
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    
    best_test_loss = float('inf')
    best_train_loss = float('inf')
    epochs_no_improve = 0
    
    print("Starting training...")
    
    # Training Loop
    for epoch in tqdm(range(epochs)):
        vf.train()
        epoch_loss = 0.0
        num_batches = 0
        shuffle(train_triplets)
        
        # Iterate over triplet batches
        for i in range(0, len(train_triplets), batch_size):
            batch = train_triplets[i: i + batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            
            # Prepare batch embeddings and labels
            batch_embeddings, batch_labels = get_embeddings_labels_from_triplets(batch, train_cls_map, class2idx)

            if batch_embeddings is None or len(batch_embeddings) == 0:
                continue

            batch_embeddings = batch_embeddings.to(device).float()
            batch_labels = batch_labels.to(device).long()
            
            # Apply vector field transformation
            pred_embeddings = euler_integration(batch_embeddings, vf, steps=10)
            loss = loss_func(pred_embeddings, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Compute test loss after each epoch 
        vf.eval()
        test_loss = compute_triplet_loss(vf, test_triplets, loss_func, test_cls_map, class2idx, device, steps=10)

        avg_loss = epoch_loss / max(1, num_batches)  

        # LR Warmup and Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        if epoch >= warmup_epochs:
            scheduler.step(test_loss)
        else:
            scheduler.step(float('inf'))

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"LR reduced at epoch {epoch + 1} → {new_lr:.1e}")

        # Early stopping combining test and train loss
        if (test_loss < best_test_loss - delta) or (avg_loss < best_train_loss - delta):
            best_test_loss = min(test_loss, best_test_loss)
            best_train_loss = min(avg_loss, best_train_loss)
            epochs_no_improve = 0
            torch.save(vf.state_dict(), "best_vf.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Test Loss = {test_loss:.4f}")

    # Save final model
    torch.save(vf.state_dict(), "vf_model.pth")
    print("Training completed. Model saved as 'vf_model.pth'")
    
    return vf

def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup
    img_dir = "./data_cifar10_style_public"
    device, model, preprocess = setup_device_and_model()
    
    # Load and process data
    embeddings, cls_map, parent_child, class_to_imgs = setup_data(img_dir, device, model, preprocess)
    
    # Create train/test splits
    train_triplets, test_triplets, train_cls_map, test_cls_map = create_train_test_splits(embeddings, cls_map, parent_child)
    
    # Train model
    trained_model = train_model(train_triplets, test_triplets, train_cls_map, test_cls_map, device)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()