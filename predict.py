#!/usr/bin/env python3
"""
ENTRep 2025 Track 2 Prediction Script
Image-to-Image Retrieval with Vector Field and Reranking
"""

import os
import torch
import torch.nn.functional as F
import csv
from pathlib import Path
from tqdm import tqdm
import warnings

from models import VectorField, RerankModel
from utils import (
    setup_device_and_clip,
    normalize_embeddings,
    save_json,
    compute_accuracy_at_1,
    evaluate_retrieval_accuracy
)

def preprocess_and_embed(image_path, model, preprocess, vf, device):
    """
    Preprocess and embed an image using CLIP + Vector Field
    
    Args:
        image_path: Path to the image file
        model: CLIP model
        preprocess: CLIP preprocessing function
        vf: Vector Field model
        device: Device to run on
        
    Returns:
        torch.Tensor: Transformed embedding vector
    """
    try:
        from PIL import Image
        # Load and convert image to RGB
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Image open failed: {e}")

    # Apply CLIP preprocessing and move to device
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate image embedding using CLIP
        emb = model.encode_image(image)

        # Check that the embedding has expected shape [1, 512]
        if emb.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")

        # Normalize the embedding vector
        emb = emb / emb.norm(dim=-1, keepdim=True)

        # Ensure correct dtype before passing to vector field
        emb = emb.to(next(vf.parameters()).dtype)

        # Apply learned vector field at time t = 0.0
        emb_vf = vf(emb, t=torch.tensor([[0.0]], device=emb.device)).squeeze(0).cpu()

        # Ensure the output is a 1D vector
        if emb_vf.ndim != 1:
            raise RuntimeError(f"VectorField output is not 1D: {emb_vf.shape}")

    return emb_vf

def rerank_topk_by_class(candidate_names, candidate_scores, predictor, models, img_dir, bonus=0.01):
    """
    Re-rank a list of top-k image candidates by promoting those that share the dominant class.

    Args:
        candidate_names (List[str]): Top-k image file names to be re-ranked.
        candidate_scores (List[float]): Corresponding similarity scores.
        predictor (RerankModel): Classifier wrapper to load and use ensemble models.
        models (List): List of loaded classification models.
        img_dir (str): Path to image directory.
        bonus (float): Bonus score to add for images in the dominant class.

    Returns:
        reranked_names (List[str]): Candidate names sorted by adjusted score.
    """
    name_to_class = {}   # Map image name → predicted class
    class_votes = {}     # Count class frequency

    # Predict class for each candidate
    for img_name in candidate_names:
        img_path = os.path.join(img_dir, img_name)
        pred_class = predictor.ensemble_predict_batch(models, [img_path], use_tta=False)[img_name]
        name_to_class[img_name] = pred_class
        class_votes[pred_class] = class_votes.get(pred_class, 0) + 1

    # Identify majority class
    main_class = max(class_votes, key=class_votes.get)

    # Apply bonus score for images in majority class
    reranked = []
    for name, score in zip(candidate_names, candidate_scores):
        bonus_score = bonus if name_to_class[name] == main_class else 0.0
        reranked.append((name, score + bonus_score))

    # Sort descending by score
    reranked = sorted(reranked, key=lambda x: -x[1])
    reranked_names = [name for name, _ in reranked]
    return reranked_names

def load_image_list(csv_path):
    """Load image names from CSV file"""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        image_list = [row[0].strip() for row in reader if row]
    print(f"Loaded {len(image_list)} images from {csv_path}")
    return image_list

def compute_embeddings(image_list, image_dir, model, preprocess, vf, device):
    """Compute embeddings for all images"""
    print("Computing embeddings for all images...")
    all_embeddings = {}

    for img_name in tqdm(image_list, desc="Embedding images"):
        img_path = os.path.join(image_dir, img_name)
        try:
            emb = preprocess_and_embed(img_path, model, preprocess, vf, device)
            all_embeddings[img_name] = emb
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")
    
    print(f"Successfully embedded {len(all_embeddings)} images")
    return all_embeddings

def prepare_embeddings_tensor(all_embeddings):
    """Filter valid embeddings and create tensor"""
    valid_img_names = []
    valid_embeddings = []

    for name in all_embeddings:
        emb = all_embeddings[name]
        # Ensure embedding is a 1D torch.Tensor
        if isinstance(emb, torch.Tensor) and emb.ndim == 1:
            valid_img_names.append(name)
            valid_embeddings.append(emb)
        else:
            print(f"⚠️ Invalid embedding: {name} → {type(emb)}, shape = {getattr(emb, 'shape', None)}")

    if not valid_embeddings:
        raise ValueError("❌ No valid embeddings found!")

    # Stack embeddings into a tensor and normalize
    img_names = valid_img_names
    embeddings = torch.stack(valid_embeddings)  # Shape: [N, D]
    embeddings = normalize_embeddings(embeddings)  # Cosine-normalized embeddings
    
    print(f"Created embeddings tensor: {embeddings.shape}")
    return img_names, embeddings

def perform_retrieval(img_names, embeddings, predictor, models, image_dir, top_k=5, bonus=0.01):
    """Perform image retrieval with class-based reranking"""
    print("Performing image retrieval with reranking...")
    retrieval_results = {}
    retrieval_results_top_k = {}  # Store top-k for accuracy evaluation

    for i, query_name in enumerate(tqdm(img_names, desc="Retrieving similar images")):
        query_emb = embeddings[i].unsqueeze(0)  # Shape: [1, D]
        # Exclude current image to avoid self-matching
        others = torch.cat([embeddings[:i], embeddings[i+1:]], dim=0)  # Shape: [N-1, D]

        # Compute cosine similarities
        sims = (others @ query_emb.T).squeeze()  # Shape: [N-1]

        # Retrieve top-k most similar images
        topk = torch.topk(sims, k=top_k)
        topk_indices = topk.indices.tolist()
        topk_scores = topk.values.tolist()

        # Map indices to actual image names (adjust index if skipped self)
        candidate_names = []
        candidate_scores = []
        for j, idx in enumerate(topk_indices):
            idx_adjusted = idx if idx < i else idx + 1
            candidate_names.append(img_names[idx_adjusted])
            candidate_scores.append(topk_scores[j])

        # Store top-k results for evaluation
        retrieval_results_top_k[query_name] = candidate_names

        # Re-rank candidates using class prediction
        reranked = rerank_topk_by_class(
            candidate_names, candidate_scores,
            predictor, models, image_dir, bonus=bonus
        )

        # Save only the top-1 match
        retrieval_results[query_name] = reranked[0]

    print(f"Retrieval completed for {len(retrieval_results)} images")
    return retrieval_results, retrieval_results_top_k

def evaluate_retrieval_performance(retrieval_results_top_k, ground_truth_file=None):
    """
    Evaluate retrieval performance if ground truth is available
    
    Args:
        retrieval_results_top_k: Dictionary of query -> [top-k results]
        ground_truth_file: Path to ground truth file (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    if ground_truth_file is None or not os.path.exists(ground_truth_file):
        print("No ground truth file provided or file not found. Skipping accuracy evaluation.")
        return None
    
    try:
        # Load ground truth
        import json
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Evaluate retrieval accuracy
        metrics = evaluate_retrieval_accuracy(retrieval_results_top_k, ground_truth, k_values=[1, 3, 5])
        
        print("\n=== Retrieval Performance Metrics ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating retrieval performance: {e}")
        return None

def main():
    """Main prediction pipeline"""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Configuration
    image_dir = "./ENTRep_Private_Dataset_update/imgs"
    csv_path = "i2i.csv"
    rerank_model_dir = "./convnextbase-ensemble-metalearner"
    output_json = "rerank003.json"
    
    print("=== ENTRep 2025 Track 2 Prediction Pipeline ===")
    
    # Step 1: Setup models
    print("\n1. Setting up models...")
    device, model, preprocess = setup_device_and_clip()
    
    # Load Vector Field model
    embed_dim = 512
    vf = VectorField(embed_dim).to(device).float()
    vf.load_state_dict(torch.load("./vf_model.pth", map_location=device))
    vf.eval()
    
    # Step 2: Initialize reranking model
    print("\n2. Loading reranking models...")
    predictor = RerankModel(model_dir=rerank_model_dir)
    models = predictor.load_ensemble_models()
    
    # Step 3: Load image list
    print("\n3. Loading image list...")
    image_list = load_image_list(csv_path)
    
    # Step 4: Compute embeddings
    print("\n4. Computing embeddings...")
    all_embeddings = compute_embeddings(image_list, image_dir, model, preprocess, vf, device)
    
    # Step 5: Prepare embeddings tensor
    print("\n5. Preparing embeddings tensor...")
    img_names, embeddings = prepare_embeddings_tensor(all_embeddings)
    
    # Step 6: Perform retrieval with reranking
    print("\n6. Performing retrieval...")
    retrieval_results, retrieval_results_top_k = perform_retrieval(img_names, embeddings, predictor, models, image_dir)
    
    # Step 7: Evaluate performance (if ground truth available)
    print("\n7. Evaluating performance...")
    ground_truth_file = "ground_truth.json"  # Optional ground truth file
    metrics = evaluate_retrieval_performance(retrieval_results_top_k, ground_truth_file)
    
    # Step 8: Save results
    print("\n8. Saving results...")
    save_json(retrieval_results, output_json)
    
    # Save additional metrics if available
    if metrics:
        metrics_file = output_json.replace('.json', '_metrics.json')
        save_json(metrics, metrics_file)
    
    print("\n=== Prediction pipeline completed successfully! ===")
    
    # Print some sample results
    print(f"\nSample results (first 5):")
    for i, (query, result) in enumerate(list(retrieval_results.items())[:5]):
        print(f"  {query} → {result}")
    
    # Print final summary
    if metrics:
        print(f"\nFinal Performance Summary:")
        print(f"  ACC@1: {metrics.get('accuracy@1', 'N/A'):.4f}")
        print(f"  ACC@3: {metrics.get('accuracy@3', 'N/A'):.4f}")
        print(f"  ACC@5: {metrics.get('accuracy@5', 'N/A'):.4f}")

if __name__ == "__main__":
    main()