"""
Prediction utilities for ENTRep 2025 Track 2
Helper functions for inference and similarity calculations
"""

import torch
import torch.nn.functional as F
import numpy as np
import json

def calculate_similarity_matrix(embeddings):
    """
    Calculate cosine similarity matrix for embeddings
    
    Args:
        embeddings: Tensor of shape [N, D]
        
    Returns:
        torch.Tensor: Similarity matrix of shape [N, N]
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    return similarity_matrix

def get_top_k_similar(query_embedding, database_embeddings, k=5, exclude_self=True):
    """
    Get top-k most similar embeddings from database
    
    Args:
        query_embedding: Query embedding tensor [D]
        database_embeddings: Database embeddings tensor [N, D]
        k: Number of top results to return
        exclude_self: Whether to exclude exact matches
        
    Returns:
        tuple: (indices, similarities) of top-k results
    """
    query_embedding = F.normalize(query_embedding.unsqueeze(0), dim=1)
    database_embeddings = F.normalize(database_embeddings, dim=1)
    
    similarities = torch.mm(query_embedding, database_embeddings.t()).squeeze()
    
    if exclude_self:
        # Set diagonal to very low value to exclude self-matches
        similarities[similarities > 0.999] = -1
    
    top_k_values, top_k_indices = torch.topk(similarities, k)
    
    return top_k_indices, top_k_values

def create_submission_format(predictions, output_path):
    """
    Create submission file in the required format
    
    Args:
        predictions: Dictionary of query -> result mappings
        output_path: Path to save the submission file
    """
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Submission file saved to: {output_path}")

def compute_embedding_statistics(embeddings):
    """
    Compute statistics for embeddings
    
    Args:
        embeddings: Tensor of embeddings [N, D]
        
    Returns:
        dict: Statistics including mean, std, etc.
    """
    stats = {
        'mean_norm': torch.norm(embeddings, dim=1).mean().item(),
        'std_norm': torch.norm(embeddings, dim=1).std().item(),
        'embedding_dim': embeddings.shape[1],
        'num_embeddings': embeddings.shape[0]
    }
    return stats

def filter_predictions_by_confidence(predictions, confidence_threshold=0.5):
    """
    Filter predictions based on confidence threshold
    
    Args:
        predictions: Dictionary of predictions with confidence scores
        confidence_threshold: Minimum confidence to keep prediction
        
    Returns:
        dict: Filtered predictions
    """
    filtered = {}
    for key, value in predictions.items():
        if isinstance(value, dict) and 'confidence' in value:
            if value['confidence'] >= confidence_threshold:
                filtered[key] = value
        else:
            # If no confidence score, keep the prediction
            filtered[key] = value
    
    return filtered

# Example usage and testing
if __name__ == "__main__":
    print("Testing prediction utilities...")
    
    # Test basic functionality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test similarity calculation
    test_embeddings = torch.randn(10, 512)
    sim_matrix = calculate_similarity_matrix(test_embeddings)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # Test top-k retrieval
    query_emb = torch.randn(512)
    db_embs = torch.randn(100, 512)
    indices, similarities = get_top_k_similar(query_emb, db_embs, k=5)
    print(f"Top-5 indices: {indices}")
    print(f"Top-5 similarities: {similarities}")
    
    # Test embedding statistics
    stats = compute_embedding_statistics(test_embeddings)
    print(f"Embedding stats: {stats}")
    
    print("Prediction utilities ready for use!")