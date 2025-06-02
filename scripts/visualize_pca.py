#!/usr/bin/env python3
"""
Visualize and test PCA models created by out_precompute_pca.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.decomposition import PCA

def load_pca_model(object_name: str, model_name: str = "dinov2_vitl14", pca_dir: str = "pca_model"):
    """Load a PCA model for the given object."""
    pca_path = Path(pca_dir) / f"{object_name}_{model_name}.pkl"
    
    if not pca_path.exists():
        print(f"PCA model not found: {pca_path}")
        return None
    
    try:
        with open(pca_path, 'rb') as f:
            pca_model_dict = pickle.load(f)
        return pca_model_dict
    except Exception as e:
        print(f"Failed to load PCA model: {e}")
        return None

def visualize_pca_components(pca_model_dict, object_name):
    """Visualize PCA components and statistics."""
    if pca_model_dict is None:
        return
    
    pca = pca_model_dict['pca']
    explained_variance = pca_model_dict['explained_variance_ratio']
    num_samples = pca_model_dict['num_samples']
    
    print(f"\n=== PCA Model for {object_name} ===")
    print(f"Number of samples: {num_samples}")
    print(f"Feature dimension: {pca_model_dict['feature_dim']}")
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total explained variance: {explained_variance.sum():.3f}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(3), explained_variance)
    plt.title(f'{object_name}: Explained Variance')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    
    # Plot cumulative explained variance
    plt.subplot(1, 3, 2)
    plt.plot(range(1, 4), np.cumsum(explained_variance), 'bo-')
    plt.title(f'{object_name}: Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    # Show PCA components as heatmap (first few dimensions)
    plt.subplot(1, 3, 3)
    components_to_show = min(64, pca.components_.shape[1])
    plt.imshow(pca.components_[:, :components_to_show], aspect='auto', cmap='RdBu_r')
    plt.title(f'{object_name}: PCA Components (first {components_to_show} dims)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('PC Component')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"pca_analysis_{object_name}.png", dpi=150, bbox_inches='tight')
    plt.show()

def test_pca_transform(pca_model_dict, object_name):
    """Test PCA transformation with dummy features."""
    if pca_model_dict is None:
        return
    
    pca = pca_model_dict['pca']
    feat_dim = pca_model_dict['feature_dim']
    
    # Generate some dummy features
    dummy_features = np.random.randn(100, feat_dim)
    
    # Transform using PCA
    pca_features = pca.transform(dummy_features)
    
    print(f"\nTesting PCA transformation for {object_name}:")
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {pca_features.shape}")
    print(f"Output range: [{pca_features.min():.3f}, {pca_features.max():.3f}]")
    
    # Visualize transformed features
    fig = plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(pca_features[:, i], bins=20, alpha=0.7)
        plt.title(f'{object_name}: PC{i+1} Distribution')
        plt.xlabel(f'PC{i+1} Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"pca_transform_test_{object_name}.png", dpi=150, bbox_inches='tight')
    plt.show()

def analyze_all_pca_models(pca_dir: str = "pca_model", model_name: str = "dinov2_vitl14"):
    """Analyze all available PCA models."""
    pca_dir = Path(pca_dir)
    
    if not pca_dir.exists():
        print(f"PCA directory not found: {pca_dir}")
        return
    
    # Load results summary
    results_file = pca_dir / f"pca_results_{model_name}.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("=== PCA Models Summary ===")
        successful_objects = []
        failed_objects = []
        
        for obj_name, result in results.items():
            if result['status'] == 'success':
                successful_objects.append(obj_name)
                print(f"✅ {obj_name}: {result['num_fg_features']} features, "
                      f"explained variance: {sum(result['explained_variance']):.3f}")
            else:
                failed_objects.append(obj_name)
                print(f"❌ {obj_name}: {result['reason']}")
        
        print(f"\nTotal: {len(successful_objects)} successful, {len(failed_objects)} failed")
        
        # Analyze successful models
        for obj_name in successful_objects[:3]:  # Limit to first 3 for demo
            print(f"\nAnalyzing {obj_name}...")
            pca_model_dict = load_pca_model(obj_name, model_name, pca_dir)
            visualize_pca_components(pca_model_dict, obj_name)
            test_pca_transform(pca_model_dict, obj_name)
    
    else:
        print(f"Results file not found: {results_file}")
        
        # Fallback: look for .pkl files directly
        pkl_files = list(pca_dir.glob(f"*_{model_name}.pkl"))
        print(f"Found {len(pkl_files)} PCA model files")
        
        for pkl_file in pkl_files[:3]:  # Limit to first 3
            obj_name = pkl_file.stem.replace(f"_{model_name}", "")
            print(f"\nAnalyzing {obj_name}...")
            pca_model_dict = load_pca_model(obj_name, model_name, pca_dir)
            visualize_pca_components(pca_model_dict, obj_name)

def compare_pca_models(object_names: list, pca_dir: str = "pca_model", model_name: str = "dinov2_vitl14"):
    """Compare multiple PCA models."""
    models = {}
    
    for obj_name in object_names:
        model_dict = load_pca_model(obj_name, model_name, pca_dir)
        if model_dict is not None:
            models[obj_name] = model_dict
    
    if not models:
        print("No valid models found for comparison")
        return
    
    # Compare explained variance
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for obj_name, model_dict in models.items():
        explained_var = model_dict['explained_variance_ratio']
        plt.plot(range(3), explained_var, 'o-', label=obj_name)
    plt.title('Explained Variance by Component')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for obj_name, model_dict in models.items():
        explained_var = model_dict['explained_variance_ratio']
        plt.plot(range(1, 4), np.cumsum(explained_var), 'o-', label=obj_name)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    sample_counts = [model_dict['num_samples'] for model_dict in models.values()]
    plt.bar(models.keys(), sample_counts)
    plt.title('Number of Training Samples')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    total_var = [model_dict['explained_variance_ratio'].sum() for model_dict in models.values()]
    plt.bar(models.keys(), total_var)
    plt.title('Total Explained Variance')
    plt.ylabel('Total Explained Variance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("pca_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize and analyze PCA models')
    parser.add_argument('--pca_dir', type=str, default='pca_model',
                       help='Directory containing PCA models')
    parser.add_argument('--model_name', type=str, default='dinov2_vitl14',
                       help='DINOv2 model name')
    parser.add_argument('--object_name', type=str, default=None,
                       help='Specific object to analyze (default: all)')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Objects to compare')
    
    args = parser.parse_args()
    
    if args.object_name:
        # Analyze specific object
        print(f"Analyzing PCA model for: {args.object_name}")
        pca_model_dict = load_pca_model(args.object_name, args.model_name, args.pca_dir)
        visualize_pca_components(pca_model_dict, args.object_name)
        test_pca_transform(pca_model_dict, args.object_name)
    
    elif args.compare:
        # Compare multiple objects
        print(f"Comparing PCA models for: {args.compare}")
        compare_pca_models(args.compare, args.pca_dir, args.model_name)
    
    else:
        # Analyze all available models
        print("Analyzing all available PCA models...")
        analyze_all_pca_models(args.pca_dir, args.model_name)

if __name__ == "__main__":
    main() 