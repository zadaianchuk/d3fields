#!/usr/bin/env python3
"""
Precompute PCA models for all object types in the AdaManip dataset.
This script processes images and masks to extract DINOv2 features and creates
object-specific PCA models for feature visualization.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from collections import defaultdict
import argparse
import json

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# DINOv2 model configurations
DINOV2_CONFIGS = {
    'dinov2_vits14': {'feat_dim': 384, 'patch_size': 14},
    'dinov2_vitb14': {'feat_dim': 768, 'patch_size': 14},
    'dinov2_vitl14': {'feat_dim': 1024, 'patch_size': 14},
    'dinov2_vitg14': {'feat_dim': 1536, 'patch_size': 14},
}

# Object type mapping from task names to object categories
OBJECT_TYPE_MAPPING = {
    'OpenBottle': 'bottle',
    'OpenDoor': 'door', 
    'OpenSafe': 'safe',
    'OpenCoffeeMachine': 'coffee_machine',
    'OpenWindow': 'window',
    'OpenPressureCooker': 'pressure_cooker',
    'OpenPen': 'pen',
    'OpenLamp': 'lamp',
    'OpenMicroWave': 'microwave'
}

class AdaManipPCAPrecomputer:
    def __init__(self, 
                 data_root: str = "/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields",
                 output_dir: str = "pca_model",
                 model_name: str = "dinov2_vitl14",
                 device: str = "cuda",
                 target_size: int = 224,
                 max_samples_per_object: int = 50):
        
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = device
        self.target_size = target_size
        self.max_samples_per_object = max_samples_per_object
        
        # Get model configuration
        self.model_config = DINOV2_CONFIGS[model_name]
        self.feat_dim = self.model_config['feat_dim']
        self.patch_size = self.model_config['patch_size']
        
        # Calculate patch dimensions
        self.patch_h = target_size // self.patch_size
        self.patch_w = target_size // self.patch_size
        
        print(f"Initializing with model: {model_name}")
        print(f"Feature dimension: {self.feat_dim}")
        print(f"Patch grid: {self.patch_h}x{self.patch_w}")
        
        # Load DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()
        
        # Image preprocessing transform
        self.transform = T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def collect_object_samples(self, task_name: str, target_object_id: int = None):
        """
        Collect image patches for a specific object type from all environments.
        
        Args:
            task_name: Task name (e.g., 'OpenBottle')
            target_object_id: Object ID in the mask (None to auto-detect main object)
        
        Returns:
            list: List of tuples (image_patch, mask_patch) for the object
        """
        task_dir = self.data_root / task_name
        if not task_dir.exists():
            print(f"Warning: Task directory {task_dir} not found")
            return []
        
        samples = []
        
        # Iterate through all environments
        for env_dir in sorted(task_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            
            print(f"  Processing {env_dir.name}...")
            
            # Iterate through all cameras
            for cam_dir in sorted(env_dir.iterdir()):
                if not cam_dir.is_dir() or not cam_dir.name.startswith('camera_'):
                    continue
                
                color_dir = cam_dir / 'color'
                mask_dir = cam_dir / 'masks'
                
                if not color_dir.exists() or not mask_dir.exists():
                    continue
                
                # Iterate through all frames
                for color_file in sorted(color_dir.glob('*.png')):
                    frame_idx = color_file.stem
                    mask_file = mask_dir / f"{frame_idx}.png"
                    
                    if not mask_file.exists():
                        continue
                    
                    # Load image and mask
                    try:
                        image = np.array(Image.open(color_file).convert('RGB'))
                        mask = np.array(Image.open(mask_file))
                        
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]  # Take first channel if RGB
                        
                        # Auto-detect main object ID if not specified
                        unique_ids = np.unique(mask)
                        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
                        
                        if len(unique_ids) == 0:
                            continue  # No objects in this mask
                        
                        # Use specified target_object_id or the largest non-background object
                        if target_object_id is not None:
                            if target_object_id not in unique_ids:
                                continue
                            main_object_id = target_object_id
                        else:
                            # Find the object with the largest area (most pixels)
                            object_areas = []
                            for obj_id in unique_ids:
                                area = (mask == obj_id).sum()
                                object_areas.append((area, obj_id))
                            
                            if not object_areas:
                                continue
                            
                            # Select object with largest area
                            main_object_id = max(object_areas)[1]
                        
                        # Extract object region
                        object_mask = (mask == main_object_id).astype(np.uint8)
                        
                        # Find bounding box
                        y_indices, x_indices = np.where(object_mask)
                        if len(y_indices) == 0:
                            continue
                        
                        y_min, y_max = y_indices.min(), y_indices.max()
                        x_min, x_max = x_indices.min(), x_indices.max()
                        
                        # Add padding
                        h, w = image.shape[:2]
                        padding = 20
                        y_min = max(0, y_min - padding)
                        y_max = min(h, y_max + padding)
                        x_min = max(0, x_min - padding)
                        x_max = min(w, x_max + padding)
                        
                        # Extract patch
                        image_patch = image[y_min:y_max, x_min:x_max]
                        mask_patch = object_mask[y_min:y_max, x_min:x_max]
                        
                        # Skip if patch is too small
                        if image_patch.shape[0] < 32 or image_patch.shape[1] < 32:
                            continue
                        
                        samples.append((image_patch, mask_patch))
                        
                        # Limit samples to avoid memory issues
                        if len(samples) >= self.max_samples_per_object:
                            print(f"    Reached max samples ({self.max_samples_per_object})")
                            return samples
                            
                    except Exception as e:
                        print(f"    Warning: Failed to process {color_file}: {e}")
                        continue
        
        print(f"  Collected {len(samples)} samples for {task_name}")
        return samples
    
    def extract_features_from_samples(self, samples):
        """
        Extract DINOv2 features from image samples.
        
        Args:
            samples: List of (image_patch, mask_patch) tuples
        
        Returns:
            tuple: (features, mask_features) where features are DINOv2 embeddings
        """
        if not samples:
            return np.array([]), np.array([])
        
        all_features = []
        all_mask_features = []
        
        batch_size = 8  # Process in batches to avoid memory issues
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            batch_images = []
            batch_masks = []
            
            for image_patch, mask_patch in batch_samples:
                # Convert to PIL and preprocess
                pil_image = Image.fromarray(image_patch)
                processed_image = self.transform(pil_image)
                batch_images.append(processed_image)
                
                # Resize mask to match processed image
                mask_resized = cv2.resize(mask_patch.astype(np.uint8), 
                                        (self.target_size, self.target_size), 
                                        interpolation=cv2.INTER_NEAREST)
                batch_masks.append(mask_resized)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features_dict = self.model.forward_features(batch_tensor)
                batch_features = features_dict['x_norm_patchtokens']  # [batch, num_patches, feat_dim]
            
            # Convert to numpy
            batch_features_np = batch_features.cpu().numpy()
            
            # Process each sample in the batch
            for j, mask_resized in enumerate(batch_masks):
                sample_features = batch_features_np[j]  # [num_patches, feat_dim]
                
                # Create mask for patches
                # Resize mask to patch grid size
                mask_patches = cv2.resize(mask_resized, 
                                        (self.patch_w, self.patch_h), 
                                        interpolation=cv2.INTER_NEAREST)
                mask_patches = mask_patches.flatten() > 0  # Boolean mask for foreground patches
                
                all_features.append(sample_features)
                all_mask_features.append(mask_patches)
        
        if all_features:
            # Concatenate all features
            features = np.concatenate(all_features, axis=0)  # [total_patches, feat_dim]
            mask_features = np.concatenate(all_mask_features, axis=0)  # [total_patches]
            
            print(f"    Extracted features shape: {features.shape}")
            print(f"    Foreground patches: {mask_features.sum()}/{len(mask_features)}")
            
            return features, mask_features
        else:
            return np.array([]), np.array([])
    
    def create_pca_model(self, features, mask_features, object_name: str):
        """
        Create PCA model from features.
        
        Args:
            features: Feature array [num_patches, feat_dim]
            mask_features: Boolean mask for foreground patches
            object_name: Name of the object for saving
        
        Returns:
            dict: PCA model and metadata
        """
        if len(features) == 0 or mask_features.sum() == 0:
            print(f"    Warning: No valid features for {object_name}")
            return None
        
        # Extract foreground features
        fg_features = features[mask_features]
        
        print(f"    Creating PCA from {len(fg_features)} foreground features")
        
        # Create PCA model
        pca = PCA(n_components=3)
        pca.fit(fg_features)
        
        # Transform features for visualization
        pca_features = pca.transform(fg_features)
        
        # Calculate statistics
        explained_variance = pca.explained_variance_ratio_
        
        print(f"    PCA explained variance: {explained_variance}")
        
        # Create model dictionary
        pca_model = {
            'pca': pca,
            'explained_variance_ratio': explained_variance,
            'num_samples': len(fg_features),
            'feature_dim': features.shape[1],
            'object_name': object_name,
            'model_name': self.model_name
        }
        
        return pca_model
    
    def save_pca_model(self, pca_model, object_name: str):
        """Save PCA model to disk."""
        if pca_model is None:
            return
        
        output_path = self.output_dir / f"{object_name}_{self.model_name}.pkl"
        
        with open(output_path, 'wb') as f:
            pickle.dump(pca_model, f)
        
        print(f"    Saved PCA model to {output_path}")
        
        # Also save a summary
        summary_path = self.output_dir / f"{object_name}_{self.model_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"PCA Model Summary for {object_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Feature dimension: {pca_model['feature_dim']}\n")
            f.write(f"Number of samples: {pca_model['num_samples']}\n")
            f.write(f"Explained variance ratio: {pca_model['explained_variance_ratio']}\n")
            f.write(f"Total explained variance: {pca_model['explained_variance_ratio'].sum():.3f}\n")
    
    def process_all_objects(self):
        """Process all object types in the dataset."""
        print(f"Processing all objects in {self.data_root}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        results = {}
        
        for task_name, object_name in OBJECT_TYPE_MAPPING.items():
            print(f"\nProcessing {task_name} -> {object_name}")
            print("-" * 40)
            
            try:
                # Collect samples
                samples = self.collect_object_samples(task_name, target_object_id=None)
                
                if not samples:
                    print(f"  No samples found for {object_name}")
                    results[object_name] = {'status': 'failed', 'reason': 'no_samples'}
                    continue
                
                # Extract features
                features, mask_features = self.extract_features_from_samples(samples)
                
                if len(features) == 0:
                    print(f"  No features extracted for {object_name}")
                    results[object_name] = {'status': 'failed', 'reason': 'no_features'}
                    continue
                
                # Create PCA model
                pca_model = self.create_pca_model(features, mask_features, object_name)
                
                if pca_model is None:
                    print(f"  Failed to create PCA model for {object_name}")
                    results[object_name] = {'status': 'failed', 'reason': 'pca_failed'}
                    continue
                
                # Save model
                self.save_pca_model(pca_model, object_name)
                
                results[object_name] = {
                    'status': 'success',
                    'num_samples': int(len(samples)),
                    'num_features': int(len(features)),
                    'num_fg_features': int(mask_features.sum()) if len(mask_features) > 0 else 0,
                    'explained_variance': [float(x) for x in pca_model['explained_variance_ratio']]
                }
                
                print(f"  ✓ Successfully created PCA model for {object_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to process {object_name}: {e}")
                results[object_name] = {'status': 'failed', 'reason': str(e)}
        
        # Save overall results
        results_path = self.output_dir / f"pca_results_{self.model_name}.json"
        with open(results_path, 'w') as f:
            json.dump(convert_to_json_serializable(results), f, indent=2)
        
        print("\n" + "=" * 60)
        print("🎉 PCA Precomputation Complete!")
        print("=" * 60)
        
        # Summary
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total = len(results)
        
        print(f"📊 Results Summary:")
        print(f"   ✅ Successful: {successful}/{total}")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   📄 Detailed results: {results_path}")
        
        for obj_name, result in results.items():
            if result['status'] == 'success':
                print(f"   ✓ {obj_name}: {result['num_fg_features']} features, "
                      f"explained variance: {sum(result['explained_variance']):.3f}")
            else:
                print(f"   ✗ {obj_name}: {result['reason']}")


def main():
    parser = argparse.ArgumentParser(description='Precompute PCA models for AdaManip objects')
    parser.add_argument('--data_root', type=str, 
                       default="/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields",
                       help='Root directory of adamanip_d3fields data')
    parser.add_argument('--output_dir', type=str, default='pca_model',
                       help='Output directory for PCA models')
    parser.add_argument('--model_name', type=str, default='dinov2_vitl14',
                       choices=list(DINOV2_CONFIGS.keys()),
                       help='DINOv2 model to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target image size for processing')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum samples per object type')
    parser.add_argument('--object_types', nargs='+', default=None,
                       help='Specific object types to process (default: all)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = AdaManipPCAPrecomputer(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        target_size=args.target_size,
        max_samples_per_object=args.max_samples
    )
    
    # Filter object types if specified
    if args.object_types:
        original_mapping = OBJECT_TYPE_MAPPING.copy()
        OBJECT_TYPE_MAPPING.clear()
        for task_name, obj_name in original_mapping.items():
            if obj_name in args.object_types:
                OBJECT_TYPE_MAPPING[task_name] = obj_name
        print(f"Processing only: {list(OBJECT_TYPE_MAPPING.values())}")
    
    # Process all objects
    processor.process_all_objects()


if __name__ == "__main__":
    main() 