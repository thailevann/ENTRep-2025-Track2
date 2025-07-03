"""
Common utilities for ENTRep 2025 Track 2
Shared functions used across training and prediction
"""

import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter

def setup_device_and_clip():
    """
    Setup device and load CLIP model
    
    Returns:
        tuple: (device, model, preprocess)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model (ViT-B/32)
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    return device, model, preprocess

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_class_distribution(cls_map):
    """
    Print the distribution of classes in the dataset
    
    Args:
        cls_map: Class mapping dictionary
    """
    counter = Counter(cls_map.values())
    print("Class distribution:")
    for cls, count in counter.items():
        print(f"  {cls}: {count} images")

def load_image_safely(image_path):
    """
    Safely load an image with error handling
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL.Image or None: Loaded image or None if failed
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None

def normalize_embeddings(embeddings):
    """
    L2 normalize embeddings
    
    Args:
        embeddings: Tensor of embeddings
        
    Returns:
        torch.Tensor: Normalized embeddings
    """
    return F.normalize(embeddings, dim=-1)

def cosine_similarity_batch(embeddings1, embeddings2):
    """
    Compute cosine similarity between two sets of embeddings
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [M, D]
        
    Returns:
        torch.Tensor: Similarity matrix [N, M]
    """
    embeddings1 = normalize_embeddings(embeddings1)
    embeddings2 = normalize_embeddings(embeddings2)
    
    return torch.mm(embeddings1, embeddings2.t())

def save_json(data, filepath):
    """
    Save data to JSON file with proper formatting
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to: {filepath}")

def load_json(filepath):
    """
    Load data from JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_directory(directory_path):
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)

def get_file_list(directory, extensions=('.png', '.jpg', '.jpeg')):
    """
    Get list of files with specific extensions from directory
    
    Args:
        directory: Directory to search
        extensions: Tuple of file extensions to include
        
    Returns:
        list: List of file paths
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                files.append(os.path.join(root, filename))
    return files

def validate_embeddings(embeddings, expected_dim=512):
    """
    Validate embedding tensor properties
    
    Args:
        embeddings: Tensor to validate
        expected_dim: Expected embedding dimension
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(embeddings, torch.Tensor):
        print(f"Invalid type: {type(embeddings)}")
        return False
    
    if embeddings.ndim != 2:
        print(f"Invalid dimensions: {embeddings.ndim}")
        return False
    
    if embeddings.shape[1] != expected_dim:
        print(f"Invalid embedding dimension: {embeddings.shape[1]}, expected {expected_dim}")
        return False
    
    return True

def calculate_memory_usage():
    """
    Calculate current GPU memory usage if available
    
    Returns:
        dict: Memory usage statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    else:
        return {'message': 'CUDA not available'}

# Constants
CLASS_NAMES = [
    "nose-right", "nose-left", "ear-right",
    "ear-left", "vc-open", "vc-closed", "throat"
]

CLASS_TO_IDX = {
    'ear-left': 3,
    'ear-right': 2,
    'nose-left': 1,
    'nose-right': 0,
    'throat': 6,
    'vc-closed': 4,
    'vc-open': 5
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Example usage
if __name__ == "__main__":
    print("Testing common utilities...")
    
    # Test device setup
    device, model, preprocess = setup_device_and_clip()
    print(f"Device: {device}")
    
    # Test random seeds
    set_random_seeds(42)
    print("Random seeds set")
    
    # Test memory usage
    memory_stats = calculate_memory_usage()
    print(f"Memory usage: {memory_stats}")
    
    print("Common utilities ready for use!")