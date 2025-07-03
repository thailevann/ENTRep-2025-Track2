#!/usr/bin/env python3
"""
ENTRep 2025 Track 2 Training Script
Vector Field Model with Flow Matching for Entity Recognition
"""

import os
import torch
from pytorch_metric_learning import losses
from random import shuffle
from tqdm import tqdm

from models import VectorField
from utils import (
    setup_device_and_clip,
    set_random_seeds,
    print_class_distribution,
    load_images_recursively, 
    create_cls_map, 
    generate_parent_child,
    create_triplets,
    filter_parent_child,
    split_embeddings,
    filter_cls_map,
    euler_integration,
    compute_triplet_loss,
    get_embeddings_labels_from_triplets,
    setup_optimizer_and_scheduler,
    update_learning_rate,
    check_early_stopping,
    save_checkpoint,
    log_training_progress,
    compute_classification_accuracy,
    compute_triplet_accuracy,
    CLASS_TO_IDX
)

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

def train_model(train_triplets, test_triplets, train_cls_map, test_cls_map, device):
    """Main training loop"""
    # Initialize model, loss, optimizer, scheduler
    embed_dim = 512
    vf = VectorField(embed_dim).to(device).float()
    loss_func = losses.MultiSimilarityLoss()
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(vf, lr=1e-4)
    
    # Training hyperparameters
    warmup_epochs = 20
    early_stop_patience = 40
    epochs = 200
    batch_size = 32
    delta = 5e-4
    
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
            batch_embeddings, batch_labels = get_embeddings_labels_from_triplets(batch, train_cls_map, CLASS_TO_IDX)

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

        # Compute test loss and accuracy after each epoch 
        avg_loss = epoch_loss / max(1, num_batches)
        test_loss = compute_triplet_loss(vf, test_triplets, loss_func, test_cls_map, CLASS_TO_IDX, device, steps=10)
        
        # Compute accuracy metrics (every 5 epochs to save time)
        train_acc = None
        test_acc = None
        if (epoch + 1) % 5 == 0:
            # Compute training accuracy on a subset to save time
            train_subset = train_triplets[:min(500, len(train_triplets))]
            train_acc = compute_triplet_accuracy(vf, train_subset, train_cls_map, CLASS_TO_IDX, device, steps=10)
            test_acc = compute_triplet_accuracy(vf, test_triplets, test_cls_map, CLASS_TO_IDX, device, steps=10)

        # Update learning rate
        lr_reduced = update_learning_rate(optimizer, scheduler, test_loss, epoch, warmup_epochs)
        
        # Check for improvement and early stopping
        improved = (test_loss < best_test_loss - delta) or (avg_loss < best_train_loss - delta)
        if improved:
            best_test_loss = min(test_loss, best_test_loss)
            best_train_loss = min(avg_loss, best_train_loss)
            epochs_no_improve = 0
            save_checkpoint(vf, optimizer, epoch, best_test_loss, "best_vf.pt")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Log progress with accuracy metrics
        log_training_progress(epoch, avg_loss, test_loss, optimizer.param_groups[0]['lr'], 
                            train_acc=train_acc, test_acc=test_acc)

    # Save final model
    torch.save(vf.state_dict(), "vf_model.pth")
    print("Training completed. Model saved as 'vf_model.pth'")
    
    return vf

def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Setup
    img_dir = "./data_cifar10_style_public"
    device, model, preprocess = setup_device_and_clip()
    
    # Load and process data
    embeddings, cls_map, parent_child, class_to_imgs = setup_data(img_dir, device, model, preprocess)
    
    # Create train/test splits
    train_triplets, test_triplets, train_cls_map, test_cls_map = create_train_test_splits(embeddings, cls_map, parent_child)
    
    # Train model
    trained_model = train_model(train_triplets, test_triplets, train_cls_map, test_cls_map, device)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()