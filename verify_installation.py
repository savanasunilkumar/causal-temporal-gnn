"""
Verification script to ensure the refactored code structure is working correctly.
"""

import sys

def verify_imports():
    """Verify all modules can be imported."""
    print("Verifying imports...")
    errors = []
    
    try:
        from causal_gnn.config import Config
        print("✓ Config imported successfully")
    except Exception as e:
        errors.append(f"✗ Config import failed: {e}")
    
    try:
        from causal_gnn.models.uact_gnn import CausalTemporalGNN
        print("✓ UACT-GNN model imported successfully")
    except Exception as e:
        errors.append(f"✗ UACT-GNN model import failed: {e}")
    
    try:
        from causal_gnn.models.fusion import LearnableMultiModalFusion
        print("✓ Multi-modal fusion imported successfully")
    except Exception as e:
        errors.append(f"✗ Multi-modal fusion import failed: {e}")
    
    try:
        from causal_gnn.data.processor import DataProcessor
        print("✓ Data processor imported successfully")
    except Exception as e:
        errors.append(f"✗ Data processor import failed: {e}")
    
    try:
        from causal_gnn.data.dataset import RecommendationDataset, create_dataloaders
        print("✓ Dataset classes imported successfully")
    except Exception as e:
        errors.append(f"✗ Dataset classes import failed: {e}")
    
    try:
        from causal_gnn.data.samplers import NegativeSampler
        print("✓ Negative sampler imported successfully")
    except Exception as e:
        errors.append(f"✗ Negative sampler import failed: {e}")
    
    try:
        from causal_gnn.causal.discovery import CausalGraphConstructor
        print("✓ Causal discovery imported successfully")
    except Exception as e:
        errors.append(f"✗ Causal discovery import failed: {e}")
    
    try:
        from causal_gnn.training.trainer import RecommendationSystem
        print("✓ Training system imported successfully")
    except Exception as e:
        errors.append(f"✗ Training system import failed: {e}")
    
    try:
        from causal_gnn.training.evaluator import Evaluator
        print("✓ Evaluator imported successfully")
    except Exception as e:
        errors.append(f"✗ Evaluator import failed: {e}")
    
    try:
        from causal_gnn.utils.cold_start import ColdStartSolver
        print("✓ Cold start solver imported successfully")
    except Exception as e:
        errors.append(f"✗ Cold start solver import failed: {e}")
    
    try:
        from causal_gnn.utils.checkpointing import ModelCheckpointer
        print("✓ Checkpointer imported successfully")
    except Exception as e:
        errors.append(f"✗ Checkpointer import failed: {e}")
    
    try:
        from causal_gnn.utils.logging import ExperimentLogger, setup_logging
        print("✓ Logging utilities imported successfully")
    except Exception as e:
        errors.append(f"✗ Logging utilities import failed: {e}")
    
    return errors

def verify_structure():
    """Verify directory structure."""
    import os
    print("\nVerifying directory structure...")
    
    required_dirs = [
        'causal_gnn',
        'causal_gnn/models',
        'causal_gnn/data',
        'causal_gnn/causal',
        'causal_gnn/training',
        'causal_gnn/utils',
        'causal_gnn/scripts'
    ]
    
    required_files = [
        'causal_gnn/__init__.py',
        'causal_gnn/config.py',
        'causal_gnn/models/uact_gnn.py',
        'causal_gnn/models/fusion.py',
        'causal_gnn/data/processor.py',
        'causal_gnn/data/dataset.py',
        'causal_gnn/data/samplers.py',
        'causal_gnn/causal/discovery.py',
        'causal_gnn/training/trainer.py',
        'causal_gnn/training/evaluator.py',
        'causal_gnn/utils/cold_start.py',
        'causal_gnn/utils/checkpointing.py',
        'causal_gnn/utils/logging.py',
        'requirements.txt',
        'README.md',
        'example_usage.py'
    ]
    
    errors = []
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            errors.append(f"✗ Directory missing: {dir_path}")
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            errors.append(f"✗ File missing: {file_path}")
    
    return errors

def main():
    """Run all verification checks."""
    print("="*80)
    print("Enhanced UACT-GNN Installation Verification")
    print("="*80)
    
    # Check structure
    structure_errors = verify_structure()
    
    # Check imports
    import_errors = verify_imports()
    
    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    
    all_errors = structure_errors + import_errors
    
    if not all_errors:
        print("\n✅ All checks passed! The installation is working correctly.")
        print("\nYou can now:")
        print("  1. Run the example: python example_usage.py")
        print("  2. Train a model: python causal_gnn/scripts/train.py --data_path <path>")
        print("  3. Use the Python API (see README.md)")
        return 0
    else:
        print(f"\n✗ {len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease check your installation and ensure all files are in place.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

