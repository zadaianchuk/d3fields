#!/usr/bin/env python3
"""
Test script for PCA precomputation.
This runs a quick test on a subset of objects to verify the implementation.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())

from scripts.our_precompute_pca import AdaManipPCAPrecomputer, OBJECT_TYPE_MAPPING

def test_pca_precomputation():
    """Test PCA precomputation on a subset of objects."""
    print("🧪 Testing PCA Precomputation")
    print("=" * 50)
    
    # Test with a smaller subset and fewer samples
    test_processor = AdaManipPCAPrecomputer(
        data_root="/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields",
        output_dir="test_pca_model",
        model_name="dinov2_vitl14",
        device="cuda",
        target_size=224,
        max_samples_per_object=10  # Reduced for testing
    )
    
    # Test with just one object type
    test_object_types = ['bottle']  # Start with bottle
    
    # Temporarily modify the mapping to test only bottle
    original_mapping = OBJECT_TYPE_MAPPING.copy()
    OBJECT_TYPE_MAPPING.clear()
    OBJECT_TYPE_MAPPING['OpenBottle'] = 'bottle'
    
    try:
        test_processor.process_all_objects()
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original mapping
        OBJECT_TYPE_MAPPING.clear()
        OBJECT_TYPE_MAPPING.update(original_mapping)

if __name__ == "__main__":
    test_pca_precomputation() 