#!/usr/bin/env python3
"""
Wrapper script to run PCA precomputation from the main directory.
This ensures proper imports and paths when running the PCA precomputation.
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Main function that calls the PCA precomputation script."""
    try:
        # Import from scripts directory
        from our_precompute_pca import main as pca_main
        
        # Run the PCA precomputation
        print("🚀 Running PCA Precomputation from main directory")
        print("=" * 60)
        pca_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages (torch, torchvision, etc.) are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running PCA precomputation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 