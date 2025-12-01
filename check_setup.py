#!/usr/bin/env python
"""
Setup Verification Script
Run this to check if your environment is properly configured.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Sleep Detection Project - Setup Verification")
print("=" * 60)

errors = []
warnings = []

# 1. Check Python version
print("\n1. Checking Python version...")
if sys.version_info >= (3, 10):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    errors.append(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} found. Need 3.10+")

# 2. Check required packages
print("\n2. Checking required packages...")
required_packages = [
    "torch",
    "pytorch_lightning",
    "hydra",
    "polars",
    "pandas",
    "numpy",
    "wandb",
    "torchvision",
    "sklearn",
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        errors.append(f"   ✗ {package} not installed")

# 3. Check CUDA availability
print("\n3. Checking GPU/CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        warnings.append("   ! CUDA not available - will use CPU (slower)")
except:
    errors.append("   ✗ Cannot check CUDA (torch not installed)")

# 4. Check directory structure
print("\n4. Checking directory structure...")
project_root = Path(__file__).parent

dirs_to_check = [
    ("run", True),
    ("src", True),
    ("data", False),
    ("processed_data", False),
    ("output", False),
]

for dir_name, required in dirs_to_check:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"   ✓ {dir_name}/")
    elif required:
        errors.append(f"   ✗ {dir_name}/ not found (required)")
    else:
        warnings.append(f"   ! {dir_name}/ not found (will be created)")

# 5. Check configuration files
print("\n5. Checking configuration files...")
config_files = [
    "run/conf/train.yaml",
    "run/conf/train_dev.yaml",
    "run/conf/dir/local.yaml",
    "run/conf/split/dev_tiny.yaml",
]

for config_file in config_files:
    config_path = project_root / config_file
    if config_path.exists():
        print(f"   ✓ {config_file}")
    else:
        errors.append(f"   ✗ {config_file} not found")

# 6. Check data files
print("\n6. Checking data files...")
data_dir = project_root / "data"
if data_dir.exists():
    data_files = [
        "train_series.parquet",
        "train_events.csv",
        "test_series.parquet",
    ]

    for data_file in data_files:
        data_path = data_dir / data_file
        if data_path.exists():
            size_mb = data_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {data_file} ({size_mb:.1f} MB)")
        else:
            warnings.append(f"   ! {data_file} not found (run download step)")
else:
    warnings.append("   ! data/ directory not found (run download step)")

# 7. Check local.yaml paths
print("\n7. Checking local.yaml paths...")
local_yaml_path = project_root / "run/conf/dir/local.yaml"
if local_yaml_path.exists():
    try:
        import yaml
        with open(local_yaml_path) as f:
            config = yaml.safe_load(f)

        for key, path in config.items():
            if key == "sub_dir":
                continue
            path_obj = Path(path)
            if path_obj.is_absolute():
                print(f"   ✓ {key}: {path}")
            else:
                warnings.append(f"   ! {key} uses relative path: {path}")
    except Exception as e:
        errors.append(f"   ✗ Error reading local.yaml: {e}")
else:
    errors.append("   ✗ local.yaml not found")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if not errors and not warnings:
    print("\n✓ All checks passed! You're ready to run the project.")
    print("\nNext step:")
    print("  python run/prepare_dev.py dir=local")
elif not errors:
    print(f"\n✓ Setup is mostly complete ({len(warnings)} warnings)")
    print("\nWarnings:")
    for warning in warnings:
        print(warning)
    print("\nYou can proceed, but some features may not work optimally.")
else:
    print(f"\n✗ Setup incomplete ({len(errors)} errors, {len(warnings)} warnings)")
    print("\nErrors:")
    for error in errors:
        print(error)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(warning)
    print("\nPlease fix the errors before running the project.")
    print("See SETUP_GUIDE.md for detailed instructions.")

print("\n" + "=" * 60)

# Exit code
sys.exit(1 if errors else 0)
