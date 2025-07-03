"""
Reranking models for ENTRep 2025 Track 2
Ensemble model for reranking image retrieval results based on class predictions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import json
import csv
from pathlib import Path
import timm
import pickle
import warnings
import random

class RerankModel:
    """
    Ensemble model for reranking image retrieval results based on class predictions
    """
    
    def __init__(self, model_dir, device='cuda'):
        """
        Initialize the reranking model
        
        Args:
            model_dir: Directory containing the ensemble models
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.class_names = [
            "nose-right", "nose-left", "ear-right",
            "ear-left", "vc-open", "vc-closed", "throat"
        ]
        self.num_classes = len(self.class_names)

        print(f"Using device: {self.device}")

    def load_ensemble_models(self):
        """
        Load all models from the ensemble directory
        
        Returns:
            list: List of model dictionaries with model, weight, and name
        """
        print("Loading ensemble models...")

        with open(self.model_dir / 'ensemble_info.pkl', 'rb') as f:
            ensemble_info = pickle.load(f)

        models = []
        model_names = ensemble_info['models']
        weights = ensemble_info['weights']

        for i, model_name in enumerate(model_names):
            print(f"Loading model {i+1}/{len(model_names)}: {model_name}")

            if 'convnext' in model_name.lower():
                base_name = "convnext_base.fb_in22k_ft_in1k"
            elif 'efficientnet' in model_name.lower():
                base_name = "efficientnet_b4"
            else:
                base_name = "convnext_base.fb_in22k_ft_in1k"

            model = timm.create_model(base_name, pretrained=False, num_classes=self.num_classes)
            state_dict = torch.load(self.model_dir / f"ensemble_model_{i}.pt", map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            models.append({'model': model, 'weight': weights[i], 'name': model_name})

        print(f"Loaded {len(models)} models successfully.")
        return models

    def load_test_data(self, csv_path, img_dir):
        """
        Load test image paths from CSV file
        
        Args:
            csv_path: Path to CSV file containing image names
            img_dir: Directory containing the images
            
        Returns:
            list: List of valid image file paths
        """
        test_files = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    img_name = row[0].strip()
                    img_path = Path(img_dir) / img_name
                    if img_path.exists():
                        test_files.append(str(img_path))
                    else:
                        print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(test_files)} test images.")
        return test_files

    def get_tta_transforms(self, img_size=224, n_aug=5):
        """
        Generate a list of transforms for Test Time Augmentation (TTA)
        
        Args:
            img_size: Target image size
            n_aug: Number of augmentations to generate
            
        Returns:
            list: List of transform compositions
        """
        class ResizeOrPad:
            def __init__(self, min_size):
                self.min_size = min_size

            def __call__(self, img):
                w, h = img.size
                if w < self.min_size or h < self.min_size:
                    scale = self.min_size / min(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    return T.functional.resize(img, (new_h, new_w))
                return img

        # Base transform (no augmentation)
        transforms = [
            T.Compose([
                ResizeOrPad(img_size),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        ]

        # Add augmented transforms
        for i in range(n_aug - 1):
            resize_delta = random.choice([-20, -10, 0, 10, 20])
            target_size = max(img_size, img_size + resize_delta)

            transforms.append(
                T.Compose([
                    ResizeOrPad(target_size + 20),
                    T.Resize((target_size + 10, target_size + 10)),
                    T.CenterCrop(img_size) if i % 2 == 0 else T.RandomCrop(img_size),
                    T.RandomApply([
                        T.ColorJitter(
                            brightness=random.uniform(0.1, 0.2),
                            contrast=random.uniform(0.1, 0.2),
                            saturation=random.uniform(0.05, 0.15),
                            hue=random.uniform(0.02, 0.05)
                        )
                    ], p=0.8),
                    T.RandomChoice([
                        T.GaussianBlur(3, sigma=(0.1, 0.5)),
                        nn.Identity()
                    ]),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            )
        return transforms

    def predict_single_image_tta(self, model, img_path, n_aug=5):
        """
        Predict a single image using Test Time Augmentation (TTA)
        
        Args:
            model: PyTorch model for prediction
            img_path: Path to the image file
            n_aug: Number of augmentations to use
            
        Returns:
            numpy.ndarray: Averaged prediction probabilities
        """
        image = Image.open(img_path).convert('RGB')
        tta_transforms = self.get_tta_transforms(n_aug=n_aug)

        aug_probs = []
        with torch.no_grad():
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                aug_probs.append(probs)

        # Weight the augmentations (base transform gets higher weight)
        weights = torch.tensor([1.5] + [1.0] * (n_aug - 1)).to(self.device)
        weights = weights / weights.sum()

        final_probs = torch.zeros_like(aug_probs[0])
        for i, probs in enumerate(aug_probs):
            final_probs += probs * weights[i]

        return final_probs.cpu().numpy().squeeze()

    def ensemble_predict_batch(self, models, test_files, use_tta=True, batch_size=32):
        """
        Predict a batch of images by ensembling multiple models
        
        Args:
            models: List of model dictionaries
            test_files: List of image file paths
            use_tta: Whether to use test time augmentation
            batch_size: Batch size for processing (not used in current implementation)
            
        Returns:
            dict: Dictionary mapping image names to predicted class indices
        """
        predictions = {}
        
        for img_path in test_files:
            img_name = Path(img_path).name
            all_model_probs = []
            model_weights = []

            for model_info in models:
                model = model_info['model']
                weight = model_info['weight']

                if use_tta:
                    probs = self.predict_single_image_tta(model, img_path, n_aug=3)
                else:
                    # Simple prediction without TTA
                    transform = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy().squeeze()

                all_model_probs.append(probs)
                model_weights.append(weight ** 2)  # Square weights for more emphasis

            # Normalize model weights
            model_weights = np.array(model_weights)
            model_weights /= model_weights.sum()

            # Weighted ensemble averaging
            final_probs = np.zeros_like(all_model_probs[0])
            for i, probs in enumerate(all_model_probs):
                final_probs += probs * model_weights[i]

            pred_class = np.argmax(final_probs)
            predictions[img_name] = int(pred_class)

        return predictions

    def predict_and_save(self, csv_path, img_dir, output_path, use_tta=True):
        """
        Run predictions and save the results to a JSON file
        
        Args:
            csv_path: Path to CSV file with image names
            img_dir: Directory containing images
            output_path: Path to save results JSON
            use_tta: Whether to use test time augmentation
            
        Returns:
            dict: Dictionary of predictions
        """
        models = self.load_ensemble_models()
        test_files = self.load_test_data(csv_path, img_dir)

        print("\nStarting prediction...")
        predictions = self.ensemble_predict_batch(models, test_files, use_tta=use_tta)

        print(f"\nSaving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved {len(predictions)} predictions")

        print("\nSample predictions:")
        for i, (img_name, pred) in enumerate(list(predictions.items())[:5]):
            print(f"  {img_name}: {pred} ({self.class_names[pred]})")

        return predictions