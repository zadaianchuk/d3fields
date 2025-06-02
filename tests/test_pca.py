#!/usr/bin/env python3
"""
Test wrapper script to run PCA test from the main directory.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Main function that calls the PCA test script."""
    try:
        # Import from scripts directory
        from test_pca_precompute import test_pca_precomputation
        
        # Run the test
        print("🧪 Running PCA Test from main directory")
        print("=" * 50)
        test_pca_precomputation()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running PCA test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 