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

def compute_recall_at_k(predictions, ground_truth, k=1):
    """
    Compute Recall@K metric for retrieval tasks
    
    Args:
        predictions: Dictionary or tensor of predictions
        ground_truth: Dictionary or tensor of ground truth labels
        k: Number of top predictions to consider
        
    Returns:
        float: Recall@K score (0.0 to 1.0)
    """
    if isinstance(predictions, dict) and isinstance(ground_truth, dict):
        # Handle dictionary format (query_id -> prediction/truth)
        correct = 0
        total = 0
        
        for query_id in predictions:
            if query_id in ground_truth:
                pred = predictions[query_id]
                truth = ground_truth[query_id]
                
                if isinstance(pred, list):
                    # Multiple predictions - check if truth is in top-k
                    top_k_preds = pred[:k] if len(pred) >= k else pred
                    if truth in top_k_preds:
                        correct += 1
                else:
                    # Single prediction - only valid for k=1
                    if k == 1 and pred == truth:
                        correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    elif torch.is_tensor(predictions) and torch.is_tensor(ground_truth):
        # Handle tensor format
        if predictions.dim() == 1:
            # Single predictions per sample
            if k == 1:
                correct = (predictions == ground_truth).float().sum()
                return correct.item() / len(ground_truth)
            else:
                raise ValueError("k > 1 requires predictions to be 2D tensor with multiple predictions per sample")
        
        elif predictions.dim() == 2:
            # Multiple predictions per sample [N, num_predictions]
            batch_size = ground_truth.size(0)
            correct = 0
            
            for i in range(batch_size):
                top_k_preds = predictions[i][:k]
                if ground_truth[i] in top_k_preds:
                    correct += 1
            
            return correct / batch_size
    
    else:
        raise ValueError("Predictions and ground_truth must both be dicts or both be tensors")

def compute_recall_at_1(predictions, ground_truth):
    """
    Compute Recall@1 metric (top-1 recall)
    
    Args:
        predictions: Dictionary or tensor of predictions
        ground_truth: Dictionary or tensor of ground truth labels
        
    Returns:
        float: Recall@1 score (0.0 to 1.0)
    """
    return compute_recall_at_k(predictions, ground_truth, k=1)

def compute_accuracy_at_k(predictions, ground_truth, k=1):
    """
    Compute Accuracy@K metric for classification tasks
    
    Args:
        predictions: Dictionary or tensor of predictions
        ground_truth: Dictionary or tensor of ground truth labels
        k: Number of top predictions to consider
        
    Returns:
        float: Accuracy@K score (0.0 to 1.0)
    """
    return compute_recall_at_k(predictions, ground_truth, k)

def compute_accuracy_at_1(predictions, ground_truth):
    """
    Compute Accuracy@1 metric (top-1 accuracy) for classification tasks
    
    Args:
        predictions: Dictionary or tensor of predictions
        ground_truth: Dictionary or tensor of ground truth labels
        
    Returns:
        float: Accuracy@1 score (0.0 to 1.0)
    """
    return compute_recall_at_k(predictions, ground_truth, k=1)

def evaluate_retrieval_recall(retrieval_results, ground_truth, k_values=[1, 5, 10]):
    """
    Evaluate retrieval recall at multiple k values
    
    Args:
        retrieval_results: Dictionary mapping query_id -> [list of retrieved items]
        ground_truth: Dictionary mapping query_id -> correct item
        k_values: List of k values to evaluate
        
    Returns:
        dict: Dictionary with recall@k for each k value
    """
    metrics = {}
    
    for k in k_values:
        recall_at_k = compute_recall_at_k(retrieval_results, ground_truth, k=k)
        metrics[f'recall@{k}'] = recall_at_k
    
    return metrics

def evaluate_retrieval_accuracy(retrieval_results, ground_truth, k_values=[1, 5, 10]):
    """
    Evaluate retrieval accuracy at multiple k values (kept for backward compatibility)
    Note: For retrieval tasks, consider using evaluate_retrieval_recall instead
    
    Args:
        retrieval_results: Dictionary mapping query_id -> [list of retrieved items]
        ground_truth: Dictionary mapping query_id -> correct item
        k_values: List of k values to evaluate
        
    Returns:
        dict: Dictionary with accuracy@k for each k value
    """
    return evaluate_retrieval_recall(retrieval_results, ground_truth, k_values)

def compute_classification_accuracy_from_embeddings(embeddings, labels, classifier_func):
    """
    Compute classification accuracy from embeddings
    
    Args:
        embeddings: Tensor of embeddings [N, D]
        labels: Tensor of true labels [N]
        classifier_func: Function that takes embeddings and returns predictions
        
    Returns:
        float: Classification accuracy
    """
    with torch.no_grad():
        predictions = classifier_func(embeddings)
        if predictions.dim() == 2:
            # If predictions are logits/probabilities, get argmax
            predictions = torch.argmax(predictions, dim=1)
        
        accuracy = (predictions == labels).float().mean().item()
        return accuracy

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