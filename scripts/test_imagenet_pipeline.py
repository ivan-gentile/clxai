#!/usr/bin/env python3
"""Test ImageNet SCL pipeline components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Testing ImageNet SCL Pipeline")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.models.imagenet_resnet import get_imagenet_resnet, ResNet152ImageNetEncoder
    from src.utils.imagenet_data import get_imagenet_loaders, get_imagenet_stats
    from src.training.losses import SupConLossV2
    print("   OK: All modules imported successfully")
except ImportError as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 2: Model creation
print("\n2. Testing model creation...")
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = get_imagenet_resnet('resnet152_imagenet', 1000, True, 128)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ResNet-152 ImageNet Encoder: {params/1e6:.1f}M params")
    print(f"   Feature dim: {model.feature_dim}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        emb = model(x)
    print(f"   Forward pass: input {x.shape} -> output {emb.shape}")
    print(f"   Embedding norm: {emb.norm(dim=1).mean():.4f} (should be ~1.0)")
    print("   OK: Model works correctly")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback; traceback.print_exc()

# Test 3: ImageNet data loading
print("\n3. Testing ImageNet data loading...")
try:
    data_dir = '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'
    print(f"   Data dir: {data_dir}")
    
    # Check if data exists
    train_path = Path(data_dir) / 'train'
    val_path = Path(data_dir) / 'validation'
    
    if not train_path.exists():
        print(f"   WARNING: Train data not found at {train_path}")
    else:
        print(f"   Train data found")
        
    if not val_path.exists():
        print(f"   WARNING: Validation data not found at {val_path}")
    else:
        print(f"   Validation data found")
    
    # Try loading a small batch
    if train_path.exists() and val_path.exists():
        train_loader, val_loader = get_imagenet_loaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Single worker for testing
            contrastive=True,
            augment=True,
            augmentation_type='none'
        )
        print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Get a sample
        images, labels = next(iter(val_loader))
        if isinstance(images, list):
            print(f"   Contrastive mode: 2 views of shape {images[0].shape}")
        else:
            print(f"   Single view: shape {images.shape}")
        print("   OK: Data loading works")
    else:
        print("   SKIP: Data not available for full test")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback; traceback.print_exc()

# Test 4: Loss function
print("\n4. Testing SupCon loss...")
try:
    import torch
    criterion = SupConLossV2(temperature=0.1)
    
    # Simulate batch with 2 views
    batch_size = 4
    emb = torch.randn(batch_size * 2, 128)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    
    loss = criterion(emb, labels)
    print(f"   SupCon loss: {loss.item():.4f}")
    print("   OK: Loss function works")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Training script exists
print("\n5. Checking training script...")
train_script = Path(__file__).parent.parent / 'src/training/train_scl_imagenet.py'
if train_script.exists():
    print(f"   OK: Training script exists at {train_script}")
else:
    print(f"   ERROR: Training script not found at {train_script}")

# Test 6: Config file
print("\n6. Checking config file...")
config_file = Path(__file__).parent.parent / 'configs/supcon_imagenet_r152.yaml'
if config_file.exists():
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print(f"   Config: {config.get('model', {}).get('architecture', 'N/A')}")
    print(f"   Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
    print(f"   Temperature: {config.get('training', {}).get('supcon', {}).get('temperature', 'N/A')}")
    print("   OK: Config file valid")
else:
    print(f"   ERROR: Config not found at {config_file}")

print("\n" + "=" * 60)
print("Pipeline test completed!")
print("=" * 60)
