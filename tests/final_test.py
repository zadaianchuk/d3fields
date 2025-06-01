#!/usr/bin/env python3
"""
Final comprehensive test of the streamlined D3Fields data loader system.
"""

import shutil
from pathlib import Path
from d3fields_data_loader import D3FieldsDataLoader, convert_to_fusion_format, run_tests, batch_convert_to_d3fields
from d3fields_utils import quick_load, list_available, get_data_stats, convert_and_save
from test_d3fields_integration import test_fusion_integration, test_original_vs_converted


def test_all_functionality():
    """Test all functionality of the streamlined system."""
    print("🧪 Final Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Core data loader
    print("\n1. Testing core data loader...")
    try:
        run_tests()  # This includes all basic functionality tests
        print("✅ Core data loader tests passed")
    except Exception as e:
        print(f"❌ Core data loader tests failed: {e}")
        raise
    
    # Test 2: Utility functions
    print("\n2. Testing utility functions...")
    try:
        # Test list_available
        print("  Testing list_available()...")
        list_available()
        
        # Test quick_load
        print("  Testing quick_load()...")
        datapoint = quick_load('OpenBottle')
        assert 'color_images' in datapoint
        assert 'depth_images' in datapoint
        assert 'camera_params' in datapoint
        
        # Test get_data_stats
        print("  Testing get_data_stats()...")
        get_data_stats('OpenBottle')
        
        # Test convert_and_save
        print("  Testing convert_and_save()...")
        test_output = "final_test_output"
        if Path(test_output).exists():
            shutil.rmtree(test_output)
        convert_and_save('OpenBottle', 'grasp_env_0', test_output)
        assert Path(test_output).exists()
        assert (Path(test_output) / "metadata.json").exists()
        shutil.rmtree(test_output)
        
        print("✅ Utility function tests passed")
        
    except Exception as e:
        print(f"❌ Utility function tests failed: {e}")
        raise
    
    # Test 3: Integration tests
    print("\n3. Testing integration...")
    try:
        # First create some converted data for testing
        print("  Generating test data...")
        batch_convert_to_d3fields("d3fields_converted", max_per_task=1, tasks=['OpenBottle'])
        
        test_fusion_integration()
        test_original_vs_converted()
        print("✅ Integration tests passed")
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        raise
    
    # Test 4: Format conversions
    print("\n4. Testing format conversions...")
    try:
        loader = D3FieldsDataLoader()
        tasks = loader.get_tasks()
        
        # Test with different tasks
        for task in tasks[:2]:  # Test first 2 tasks
            environments = loader.get_environments(task)
            
            if environments['grasp']:
                env = environments['grasp'][0]
            elif environments['manip']:
                env = environments['manip'][0]
            else:
                continue
            
            datapoint = loader.load_datapoint(task, env, frame_idx=0)
            obs = convert_to_fusion_format(datapoint)
            
            # Validate conversion
            assert obs['color'].shape[0] == len(datapoint['camera_params'])
            assert obs['depth'].shape[0] == len(datapoint['camera_params'])
            assert obs['K'].shape[0] == len(datapoint['camera_params'])
            assert obs['pose'].shape[0] == len(datapoint['camera_params'])
            
            print(f"  ✓ Tested {task}/{env}")
        
        print("✅ Format conversion tests passed")
        
    except Exception as e:
        print(f"❌ Format conversion tests failed: {e}")
        raise
    
    # Test 5: Batch operations
    print("\n5. Testing batch operations...")
    try:
        # Test small batch conversion
        batch_output = "final_batch_test"
        if Path(batch_output).exists():
            shutil.rmtree(batch_output)
        
        batch_convert_to_d3fields(batch_output, max_per_task=1, tasks=['OpenBottle'])
        
        # Verify batch output
        assert Path(batch_output).exists()
        assert (Path(batch_output) / "OpenBottle").exists()
        
        # Count converted environments
        converted_count = 0
        for env_dir in (Path(batch_output) / "OpenBottle").iterdir():
            if env_dir.is_dir() and (env_dir / "metadata.json").exists():
                converted_count += 1
        
        assert converted_count >= 1, f"Expected at least 1 converted environment, got {converted_count}"
        
        shutil.rmtree(batch_output)
        print("✅ Batch operation tests passed")
        
    except Exception as e:
        print(f"❌ Batch operation tests failed: {e}")
        raise
    
    # Test 6: Error handling
    print("\n6. Testing error handling...")
    try:
        loader = D3FieldsDataLoader()
        
        # Test invalid task
        try:
            loader.get_environments('InvalidTask')
            assert False, "Should have raised an error for invalid task"
        except AssertionError as e:
            if "Task not found" in str(e):
                pass  # Expected error
            else:
                raise
        
        # Test invalid frame index
        try:
            loader.load_datapoint('OpenBottle', 'grasp_env_0', frame_idx=999)
            assert False, "Should have raised an error for invalid frame index"
        except AssertionError as e:
            if "Frame" in str(e) and ">=" in str(e):
                pass  # Expected error
            else:
                raise
        
        print("✅ Error handling tests passed")
        
    except Exception as e:
        print(f"❌ Error handling tests failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! System is ready for use.")
    print("=" * 60)
    
    # Print usage summary
    print("\n📋 Quick Usage Summary:")
    print("─" * 30)
    print("# Load data:")
    print("from d3fields_utils import quick_load")
    print("datapoint = quick_load('OpenBottle')")
    print()
    print("# Convert for fusion:")
    print("from d3fields_data_loader import convert_to_fusion_format")
    print("obs = convert_to_fusion_format(datapoint)")
    print()
    print("# Save D3Fields format:")
    print("from d3fields_utils import convert_and_save")
    print("convert_and_save('OpenBottle', 'grasp_env_0', 'output')")
    print()
    print("🚀 System ready for D3Fields processing!")


if __name__ == "__main__":
    test_all_functionality() 