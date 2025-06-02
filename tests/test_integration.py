#!/usr/bin/env python3
"""
Integration test to verify that PCA system integration works correctly.
"""

import sys
import os
from pathlib import Path

def test_integration():
    """Test the integration between main.py and the PCA scripts."""
    print("🔧 Testing Integration")
    print("=" * 40)
    
    # Test 1: Check that OBJECT_TYPE_MAPPING is available in main.py
    try:
        # Import from main.py
        import main
        
        if hasattr(main, 'OBJECT_TYPE_MAPPING'):
            mapping = main.OBJECT_TYPE_MAPPING
            print(f"✅ OBJECT_TYPE_MAPPING found in main.py with {len(mapping)} entries")
            print(f"   Objects: {list(mapping.values())}")
        else:
            print("❌ OBJECT_TYPE_MAPPING not found in main.py")
            return False
            
    except Exception as e:
        print(f"❌ Failed to import main.py: {e}")
        return False
    
    # Test 2: Check that create_simple_pca_model function exists
    try:
        if hasattr(main, 'create_simple_pca_model'):
            print("✅ create_simple_pca_model function found in main.py")
            
            # Test calling the function with a dummy object name
            pca_model = main.create_simple_pca_model('bottle')
            if pca_model is not None:
                print("✅ PCA model creation works (fallback to dummy model)")
            else:
                print("❌ PCA model creation returned None")
                return False
        else:
            print("❌ create_simple_pca_model function not found in main.py")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test create_simple_pca_model: {e}")
        return False
    
    # Test 3: Check that scripts can be imported
    try:
        scripts_dir = Path(__file__).parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        # Try importing the PCA script
        import our_precompute_pca
        print("✅ our_precompute_pca.py can be imported from scripts/")
        
        if hasattr(our_precompute_pca, 'OBJECT_TYPE_MAPPING'):
            script_mapping = our_precompute_pca.OBJECT_TYPE_MAPPING
            print(f"✅ Script OBJECT_TYPE_MAPPING found with {len(script_mapping)} entries")
            
            # Check if mappings match
            if script_mapping == mapping:
                print("✅ Object type mappings are consistent between main.py and scripts")
            else:
                print("⚠️  Object type mappings differ between main.py and scripts")
                print(f"   Main: {mapping}")
                print(f"   Script: {script_mapping}")
        
    except Exception as e:
        print(f"❌ Failed to import scripts: {e}")
        return False
    
    # Test 4: Check wrapper scripts exist
    wrapper_scripts = ['run_pca_precompute.py', 'test_pca.py']
    for wrapper in wrapper_scripts:
        if Path(wrapper).exists():
            print(f"✅ Wrapper script {wrapper} exists")
        else:
            print(f"❌ Wrapper script {wrapper} not found")
            return False
    
    print("\n🎉 Integration test passed!")
    print("The PCA system is properly integrated.")
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1) 