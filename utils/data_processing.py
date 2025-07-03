"""
Utility functions for ENTRep 2025 Track 2
Data processing, triplet generation, and helper functions
"""

import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def load_images_recursively(folder, device, model, preprocess):
    """
    Embed all images in the training data and return them in a dictionary
    
    Args:
        folder: Path to the image folder
        device: Device to run inference on
        model: CLIP model
        preprocess: CLIP preprocessing function
        
    Returns:
        embeddings: Dictionary mapping image paths to embedding tensors
                   {
                       "img_name": embedding_tensor,
                       ...
                   }
    """
    embeddings = {}
    img_count = 0
    
    for root, dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, fn)
                try:
                    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(image)
                        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
                    rel_path = os.path.relpath(path, folder)
                    embeddings[rel_path] = emb.squeeze(0).cpu()
                    img_count += 1
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    
    print(f"Successfully embedded {img_count} images")
    return embeddings

def create_cls_map(root_dir):
    """
    Create a mapping from relative image path to its class name.

    Args:
        root_dir (str): Root directory containing class-named subfolders with images.

    Returns:
        dict: {
            "class_name/image_name.jpg": "class_name",
            ...
        }
    """
    cls_map = {}
    
    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_path = f"{cls_name}/{img_name}"
                cls_map[rel_path] = cls_name
                
    return cls_map

def generate_parent_child(cls_map, embeddings, num_children=3):
    """
    Generate (parent, top-k children) pairs based on cosine similarity within the same class.

    Args:
        cls_map (dict): Mapping from image_name → class_name
        embeddings (dict): Mapping from image_name → embedding tensor
        num_children (int): Number of most similar children to select per parent

    Returns:
        tuple: (parent_child_dict, class_to_imgs_dict)
            parent_child_dict: {
                'img_name_1': ['similar_img_1', 'similar_img_2', ...],
                ...
            }
            class_to_imgs_dict: {
                'class_name': ['img1', 'img2', ...],
                ...
            }
    """
    # Group images by class
    class_to_imgs = {}
    for img_name, cls in cls_map.items():
        class_to_imgs.setdefault(cls, []).append(img_name)

    parent_child = {}

    for cls, img_list in class_to_imgs.items():
        for img_name in img_list:
            if img_name not in embeddings:
                continue  # Skip if embedding is missing

            anchor_emb = embeddings[img_name].unsqueeze(0)  # shape: (1, D)

            # Get other images in the same class (excluding self) that have embeddings
            others = [i for i in img_list if i != img_name and i in embeddings]
            if not others:
                continue

            other_embs = torch.stack([embeddings[i] for i in others])  # shape: (N, D)

            # Compute cosine similarity between anchor and others
            sim_scores = F.cosine_similarity(anchor_emb, other_embs)  # shape: (N,)

            # Select top-k most similar images
            topk = min(num_children, len(others))
            topk_indices = torch.topk(sim_scores, k=topk).indices

            children = [others[i] for i in topk_indices.tolist()]
            parent_child[img_name] = children

    return parent_child, class_to_imgs

def get_hard_negative(anchor_name, anchor_emb, anchor_cls, class_to_imgs, embeddings, negative_map):
    """
    Select the hardest negative image (most similar but from a different class).
    Priority is given to predefined negative classes.
    
    Args:
        anchor_name: Name of the anchor image
        anchor_emb: Embedding of the anchor image
        anchor_cls: Class of the anchor image
        class_to_imgs: Dictionary mapping class names to image lists
        embeddings: Dictionary of all embeddings
        negative_map: Dictionary defining negative class relationships
        
    Returns:
        str: Name of the hard negative image, or None if not found
    """
    neg_classes = negative_map.get(anchor_cls, None)

    if not neg_classes:
        # Fallback: random other class
        neg_cls_candidates = [c for c in class_to_imgs.keys() if c != anchor_cls]
        if not neg_cls_candidates:
            return None
        neg_cls = random.choice(neg_cls_candidates)
        neg_imgs = class_to_imgs[neg_cls]
    else:
        neg_imgs = []
        for neg_cls in neg_classes:
            neg_imgs.extend(class_to_imgs.get(neg_cls, []))
        if not neg_imgs:
            return None

    anchor_emb = anchor_emb.unsqueeze(0)

    max_sim = -1
    hard_neg = None
    for neg_img in neg_imgs:
        if neg_img not in embeddings:
            continue
        neg_emb = embeddings[neg_img].unsqueeze(0)
        sim = F.cosine_similarity(anchor_emb, neg_emb).item()
        if sim > max_sim:
            max_sim = sim
            hard_neg = neg_img

    return hard_neg

def create_triplets(embeddings, cls_map, parent_child):
    """
    Create triplets (anchor, positive, negative) for training or testing.
    
    Args:
        embeddings: Dictionary of embeddings
        cls_map: Dictionary mapping image names to class names
        parent_child: Dictionary mapping parents to children
        
    Returns:
        list: List of triplets (anchor_emb, pos_emb, neg_emb, anchor_name, neg_name)
    """
    # Define class-wise incompatible (negative) relationships
    negative_map = {
        "nose-right": ["nose-left"],
        "nose-left": ["nose-right"],
        "ear-right": ["ear-left"],
        "ear-left": ["ear-right"],
        "vc-open": ["vc-closed", "throat"],
        "vc-closed": ["vc-open"],
        "throat": ["vc-open", "vc-closed"],
    }
    
    # Group images by class for negative sampling
    class_to_imgs = {}
    for img_name, cls in cls_map.items():
        class_to_imgs.setdefault(cls, []).append(img_name)
    
    triplets = []
    for anchor_name, anchor_emb in embeddings.items():
        if anchor_name not in parent_child or len(parent_child[anchor_name]) == 0:
            continue  # skip if no positive pair

        anchor_cls = cls_map[anchor_name]
        pos_name = parent_child[anchor_name][0]  # Take first (most similar) positive

        neg_name = get_hard_negative(anchor_name, anchor_emb, anchor_cls, class_to_imgs, embeddings, negative_map)
        if neg_name is None:
            continue
            
        triplets.append((
            anchor_emb,
            embeddings[pos_name],
            embeddings[neg_name],
            anchor_name,
            neg_name
        ))

    print(f"Generated {len(triplets)} triplets")
    return triplets

def filter_parent_child(parent_child_full, embeddings_subset):
    """
    Filter parent-child mapping to retain only pairs present in the current embedding subset.
    
    Args:
        parent_child_full: Full parent-child mapping
        embeddings_subset: Subset of embeddings (e.g., train or test)
        
    Returns:
        dict: Filtered parent-child mapping
    """
    parent_child_filtered = {}
    valid_imgs = set(embeddings_subset.keys())
    
    for parent, children in parent_child_full.items():
        if parent in valid_imgs:
            filtered_children = [c for c in children if c in valid_imgs]
            if filtered_children:
                parent_child_filtered[parent] = filtered_children
                
    return parent_child_filtered

def split_embeddings(embeddings, cls_map, test_size=0.2, min_images=5):
    """
    Split embeddings and cls_map into train/test sets, stratified by class.
    Classes with too few samples go entirely to the training set.
    
    Args:
        embeddings: Dictionary of all embeddings
        cls_map: Dictionary mapping image names to class names
        test_size: Fraction of data to use for testing
        min_images: Minimum number of images per class to allow splitting
        
    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    train_embeds = {}
    test_embeds = {}
    
    for cls in set(cls_map.values()):
        imgs = [img for img in cls_map if cls_map[img] == cls and img in embeddings]
        
        if len(imgs) < min_images:
            # Put all images in training set if too few samples
            for img in imgs:
                train_embeds[img] = embeddings[img]
            continue
            
        # Split stratified by class
        train_imgs, test_imgs = train_test_split(imgs, test_size=test_size, random_state=42)
        
        for img in train_imgs:
            train_embeds[img] = embeddings[img]
        for img in test_imgs:
            test_embeds[img] = embeddings[img]
            
    return train_embeds, test_embeds

def filter_cls_map(embeddings_subset, cls_map):
    """
    Filter cls_map to retain only images that are present in the given embeddings subset.
    
    Args:
        embeddings_subset: Subset of embeddings
        cls_map: Full class mapping
        
    Returns:
        dict: Filtered class mapping
    """
    return {k: v for k, v in cls_map.items() if k in embeddings_subset}

def get_triplet_batch(triplets, batch_size=32):
    """
    Sample a batch of triplets (anchor, positive, negative).
    
    Args:
        triplets: List of triplets
        batch_size: Size of the batch to sample
        
    Returns:
        tuple: (anchor_batch, positive_batch, negative_batch) or None if insufficient triplets
    """
    if len(triplets) < batch_size:
        return None
        
    selected = random.sample(triplets, batch_size)
    anchor = torch.stack([t[0] for t in selected])
    positive = torch.stack([t[1] for t in selected])
    negative = torch.stack([t[2] for t in selected])
    
    return anchor, positive, negative

def get_embeddings_labels_from_triplets(triplets_batch, cls_map, class2idx):
    """
    Convert list of triplets into tensors for model input.
    
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

def print_class_distribution(cls_map):
    """
    Print the distribution of classes in the dataset.
    
    Args:
        cls_map: Class mapping dictionary
    """
    counter = Counter(cls_map.values())
    for cls, count in counter.items():
        print(f"Class {cls}: {count} images")

def visualize_triplets(triplets, cls_map, image_dir, num_samples=5):
    """
    Visualize triplets (query, positive, negative) with matplotlib.
    
    Args:
        triplets: List of triplets
        cls_map: Class mapping dictionary
        image_dir: Directory containing images
        num_samples: Number of triplets to visualize
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if not triplets:
        print("No triplets to visualize")
        return
        
    # Sample random triplets
    sample_triplets = random.sample(triplets, min(num_samples, len(triplets)))
    
    fig, axes = plt.subplots(len(sample_triplets), 3, figsize=(12, 4 * len(sample_triplets)))
    if len(sample_triplets) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (anchor_emb, pos_emb, neg_emb, anchor_name, neg_name) in enumerate(sample_triplets):
        # Find positive name by matching embeddings
        pos_name = None
        for name, emb in embeddings.items():
            if torch.equal(emb, pos_emb):
                pos_name = name
                break
        
        if pos_name is None:
            continue
            
        # Load and display images
        try:
            images = []
            titles = []
            
            for img_name, title_prefix in [(anchor_name, "Anchor"), (pos_name, "Positive"), (neg_name, "Negative")]:
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                titles.append(f"{title_prefix}:\n{cls_map[img_name]}")
            
            for j, (img, title) in enumerate(zip(images, titles)):
                axes[i, j].imshow(img)
                axes[i, j].set_title(title, fontsize=10)
                axes[i, j].axis("off")
                
        except Exception as e:
            print(f"Error loading images for triplet {i}: {e}")
            continue
    
    plt.tight_layout()
    plt.show()

def calculate_embedding_stats(embeddings):
    """
    Calculate statistics for the embeddings.
    
    Args:
        embeddings: Dictionary of embeddings
        
    Returns:
        dict: Statistics including mean, std, min, max norms
    """
    if not embeddings:
        return {}
    
    all_embeddings = torch.stack(list(embeddings.values()))
    norms = torch.norm(all_embeddings, dim=1)
    
    stats = {
        'num_embeddings': len(embeddings),
        'embedding_dim': all_embeddings.shape[1],
        'mean_norm': norms.mean().item(),
        'std_norm': norms.std().item(),
        'min_norm': norms.min().item(),
        'max_norm': norms.max().item(),
        'mean_embedding': all_embeddings.mean(dim=0),
        'std_embedding': all_embeddings.std(dim=0)
    }
    
    return stats

# Example usage and testing
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test create_cls_map
    if os.path.exists("./data_cifar10_style_public"):
        print("Testing create_cls_map...")
        cls_map = create_cls_map("./data_cifar10_style_public")
        print(f"Created cls_map with {len(cls_map)} entries")
        
        # Print some sample entries
        sample_entries = list(cls_map.items())[:5]
        for img_path, cls_name in sample_entries:
            print(f"  {img_path} -> {cls_name}")
    
    print("Utility functions ready for use!")