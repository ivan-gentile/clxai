"""
Comprehensive smoke tests for the CLXAI codebase.
Run from the project root:
    python tests/test_all.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED, FAILED = 0, 0

def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS: {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        traceback.print_exc()
        FAILED += 1


# =====================================================================
# TEST 1: Imports
# =====================================================================
def test_imports():
    import src.models.resnet
    import src.models.imagenet_resnet
    import src.models.classifiers
    import src.training.losses
    import src.utils.data
    import src.utils.imagenet_data
    import src.evaluation.xai_evaluation
    import src.evaluation.faithfulness_imagenet50
    import src.evaluation.coherence_imagenet50

# =====================================================================
# TEST 2: ResNet-18 forward pass (CIFAR10)
# =====================================================================
def test_resnet18_forward():
    import torch
    from src.models.resnet import get_resnet18
    model = get_resnet18(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"

# =====================================================================
# TEST 3: ResNet-18 encoder (CIFAR10)
# =====================================================================
def test_resnet18_encoder():
    import torch
    from src.models.resnet import ResNet18Encoder
    enc = ResNet18Encoder(embedding_dim=128)
    x = torch.randn(4, 3, 32, 32)
    out = enc(x)
    assert out.shape == (4, 128), f"Expected (4,128), got {out.shape}"
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5), "Output should be L2-normalized"

# =====================================================================
# TEST 4: ResNet-50 forward pass (ImageNet-S50)
# =====================================================================
def test_resnet50_forward():
    import torch
    from src.models.imagenet_resnet import get_imagenet_resnet
    model = get_imagenet_resnet(num_classes=50, encoder_only=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 50), f"Expected (2,50), got {out.shape}"

# =====================================================================
# TEST 5: ResNet-50 encoder (ImageNet-S50)
# =====================================================================
def test_resnet50_encoder():
    import torch
    from src.models.imagenet_resnet import get_imagenet_resnet
    enc = get_imagenet_resnet(encoder_only=True, embedding_dim=128)
    x = torch.randn(2, 3, 224, 224)
    emb = enc(x)
    assert emb.shape == (2, 128), f"Expected (2,128), got {emb.shape}"
    raw = enc.get_embedding(x, normalize=False)
    assert raw.shape[1] == 2048, f"Raw embedding should be 2048-dim, got {raw.shape[1]}"

# =====================================================================
# TEST 6: Linear classifier
# =====================================================================
def test_linear_classifier():
    import torch
    from src.models.classifiers import LinearClassifier, train_linear_classifier
    clf = LinearClassifier(input_dim=128, num_classes=10)
    x = torch.randn(8, 128)
    out = clf(x)
    assert out.shape == (8, 10), f"Expected (8,10), got {out.shape}"

# =====================================================================
# TEST 7: SupCon loss
# =====================================================================
def test_supcon_loss():
    import torch
    from src.training.losses import SupConLoss
    loss_fn = SupConLoss(temperature=0.07)
    B = 4
    f1 = torch.nn.functional.normalize(torch.randn(B, 128), dim=1)
    f2 = torch.nn.functional.normalize(torch.randn(B, 128), dim=1)
    features = torch.cat([f1, f2], dim=0)  # (2B, D)
    labels = torch.tensor([0, 0, 1, 1])  # (B,)
    loss = loss_fn(features, labels)
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

# =====================================================================
# TEST 8: SupCon loss with two views
# =====================================================================
def test_supcon_loss_distinct_classes():
    import torch
    from src.training.losses import SupConLoss
    loss_fn = SupConLoss(temperature=0.07)
    B = 8
    f1 = torch.nn.functional.normalize(torch.randn(B, 128), dim=1)
    f2 = torch.nn.functional.normalize(torch.randn(B, 128), dim=1)
    features = torch.cat([f1, f2], dim=0)  # (2B, D)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # (B,)
    loss = loss_fn(features, labels)
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isnan(loss), "Loss should not be NaN"

# =====================================================================
# TEST 9: Triplet loss
# =====================================================================
def test_triplet_loss():
    import torch
    from src.training.losses import TripletLoss
    loss_fn = TripletLoss(margin=0.3, mining='semi-hard')
    embeddings = torch.randn(16, 512, requires_grad=True)
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss), "Loss should not be NaN"
    loss.backward()
    assert embeddings.grad is not None, "Gradients should flow"

# =====================================================================
# TEST 10: Triplet loss hard mining
# =====================================================================
def test_triplet_loss_hard():
    import torch
    from src.training.losses import TripletLoss
    loss_fn = TripletLoss(margin=0.3, mining='hard')
    embeddings = torch.randn(16, 512, requires_grad=True)
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss), "Loss should not be NaN"

# =====================================================================
# TEST 11: CIFAR10 transforms
# =====================================================================
def test_cifar10_transforms():
    import torch
    from src.utils.data import get_train_transforms, get_test_transforms, get_contrastive_transforms, TwoCropTransform
    train_t = get_train_transforms()
    test_t = get_test_transforms()
    contr_t = get_contrastive_transforms()
    two_crop = TwoCropTransform(contr_t)
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    out_train = train_t(img)
    assert out_train.shape == (3, 32, 32), f"Train transform: expected (3,32,32), got {out_train.shape}"
    out_test = test_t(img)
    assert out_test.shape == (3, 32, 32), f"Test transform: expected (3,32,32), got {out_test.shape}"
    out_single = contr_t(img)
    assert out_single.shape == (3, 32, 32), "Single contrastive transform should produce (3,32,32)"
    out_contr = two_crop(img)
    assert isinstance(out_contr, list) and len(out_contr) == 2, "TwoCropTransform should return list of 2 views"
    assert out_contr[0].shape == (3, 32, 32)

# =====================================================================
# TEST 12: ImageNet-S50 transforms
# =====================================================================
def test_imagenet_transforms():
    import torch
    from src.utils.imagenet_data import (
        get_imagenet50_train_transforms, get_imagenet50_test_transforms,
        get_imagenet50_contrastive_transforms, TwoCropTransform
    )
    train_t = get_imagenet50_train_transforms()
    test_t = get_imagenet50_test_transforms()
    contr_t = get_imagenet50_contrastive_transforms()
    two_crop = TwoCropTransform(contr_t)
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    out_train = train_t(img)
    assert out_train.shape == (3, 224, 224), f"Expected (3,224,224), got {out_train.shape}"
    out_test = test_t(img)
    assert out_test.shape == (3, 224, 224)
    out_single = contr_t(img)
    assert out_single.shape == (3, 224, 224), "Single contrastive transform should produce (3,224,224)"
    out_contr = two_crop(img)
    assert isinstance(out_contr, list) and len(out_contr) == 2

# =====================================================================
# TEST 13: TwoCropTransform
# =====================================================================
def test_two_crop_transform():
    import torch
    from src.utils.data import TwoCropTransform
    import torchvision.transforms as T
    t = TwoCropTransform(T.ToTensor())
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    out = t(img)
    assert isinstance(out, list) and len(out) == 2

# =====================================================================
# TEST 14: CE training step simulation (CIFAR10)
# =====================================================================
def test_ce_training_step_cifar10():
    import torch
    import torch.nn as nn
    from src.models.resnet import get_resnet18
    model = get_resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    model.train()
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert loss.item() > 0

# =====================================================================
# TEST 15: SCL training step simulation (CIFAR10)
# =====================================================================
def test_scl_training_step_cifar10():
    import torch
    from src.models.resnet import ResNet18Encoder
    from src.training.losses import SupConLoss
    encoder = ResNet18Encoder(embedding_dim=128)
    loss_fn = SupConLoss(temperature=0.07)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5, momentum=0.9)
    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    encoder.train()
    images = torch.cat([x1, x2], dim=0)
    features = encoder(images)  # (2B, 128)
    loss = loss_fn(features, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert loss.item() > 0

# =====================================================================
# TEST 16: Triplet training step simulation (CIFAR10)
# =====================================================================
def test_triplet_training_step_cifar10():
    import torch
    from src.models.resnet import ResNet18Encoder
    from src.training.losses import TripletLoss
    encoder = ResNet18Encoder(embedding_dim=128)
    loss_fn = TripletLoss(margin=0.3, mining='semi-hard')
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(16, 3, 32, 32)
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    encoder.train()
    emb = encoder.get_embedding(x, normalize=False)
    loss = loss_fn(emb, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =====================================================================
# TEST 17: CE training step simulation (ImageNet-S50)
# =====================================================================
def test_ce_training_step_imagenet():
    import torch
    import torch.nn as nn
    from src.models.imagenet_resnet import get_imagenet_resnet
    model = get_imagenet_resnet(num_classes=50, encoder_only=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 50, (2,))
    model.train()
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert loss.item() > 0

# =====================================================================
# TEST 18: SCL training step simulation (ImageNet-S50)
# =====================================================================
def test_scl_training_step_imagenet():
    import torch
    from src.models.imagenet_resnet import get_imagenet_resnet
    from src.training.losses import SupConLoss
    encoder = get_imagenet_resnet(encoder_only=True, embedding_dim=128)
    loss_fn = SupConLoss(temperature=0.07)
    optimizer = torch.optim.RAdam(encoder.parameters(), lr=0.001, weight_decay=1e-4)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    encoder.train()
    images = torch.cat([x1, x2], dim=0)
    features = encoder(images)  # (2B, 128)
    loss = loss_fn(features, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert loss.item() > 0

# =====================================================================
# TEST 19: Linear probe end-to-end
# =====================================================================
def test_linear_probe_training():
    import torch
    from src.models.resnet import ResNet18Encoder
    from src.models.classifiers import LinearClassifier, train_linear_classifier
    encoder = ResNet18Encoder(embedding_dim=128)
    encoder.eval()
    x = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    with torch.no_grad():
        features = encoder.get_embedding(x, normalize=True)
    clf = LinearClassifier(input_dim=512, num_classes=10)
    history = train_linear_classifier(
        clf, features, labels,
        val_embeddings=features, val_labels=labels,
        epochs=5, lr=0.1, device='cpu')
    assert 'train_acc' in history
    assert len(history['val_acc']) == 5
    assert isinstance(history['val_acc'][-1], float)

# =====================================================================
# TEST 20: Grad-CAM on ResNet-18
# =====================================================================
def test_gradcam_resnet18():
    import torch
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.models.resnet import get_resnet18
    torch.manual_seed(123)
    model = get_resnet18(num_classes=10)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    target_layers = [model.encoder.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    found_nonzero = False
    for target_cls in range(3):
        x = torch.randn(1, 3, 32, 32)
        with torch.enable_grad():
            attr = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_cls)])
        assert attr.shape == (1, 32, 32), f"Expected (1,32,32), got {attr.shape}"
        if np.max(attr) > 0:
            found_nonzero = True
            break
    assert found_nonzero, "Grad-CAM should produce non-zero maps for at least one class"
    cam.activations_and_grads.release()

# =====================================================================
# TEST 21: Grad-CAM on ResNet-50
# =====================================================================
def test_gradcam_resnet50():
    import torch
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.models.imagenet_resnet import get_imagenet_resnet
    model = get_imagenet_resnet(num_classes=50, encoder_only=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    x = torch.randn(1, 3, 224, 224)
    with torch.enable_grad():
        attr = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])
    assert attr.shape == (1, 224, 224), f"Expected (1,224,224), got {attr.shape}"
    cam.activations_and_grads.release()

# =====================================================================
# TEST 22: EigenCAM on ResNet-18
# =====================================================================
def test_eigencam_resnet18():
    import torch
    import numpy as np
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.models.resnet import get_resnet18
    model = get_resnet18(num_classes=10)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    target_layers = [model.encoder.layer4[-1]]
    cam = EigenCAM(model=model, target_layers=target_layers)
    x = torch.randn(1, 3, 32, 32)
    attr = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])
    assert attr.shape == (1, 32, 32)
    cam.activations_and_grads.release()

# =====================================================================
# TEST 23: Cosine annealing LR schedule
# =====================================================================
def test_cosine_schedule():
    import torch
    from src.models.resnet import get_resnet18
    model = get_resnet18(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for _ in range(10):
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    assert lr < 0.1, f"LR should have decreased, got {lr}"

# =====================================================================
# TEST 24: Denormalize utility
# =====================================================================
def test_denormalize():
    import torch
    from src.utils.data import denormalize, CIFAR10_MEAN, CIFAR10_STD, get_test_transforms
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    t = get_test_transforms()
    normalized = t(img).unsqueeze(0)
    recovered = denormalize(normalized)
    assert recovered.shape == normalized.shape
    assert recovered.min() >= -0.01, f"Min {recovered.min():.4f} below 0"
    assert recovered.max() <= 1.01, f"Max {recovered.max():.4f} above 1"


# =====================================================================
# RUN ALL
# =====================================================================
if __name__ == "__main__":
    tests = [
        ("All imports", test_imports),
        ("ResNet-18 forward (CIFAR10)", test_resnet18_forward),
        ("ResNet-18 encoder (CIFAR10)", test_resnet18_encoder),
        ("ResNet-50 forward (ImageNet-S50)", test_resnet50_forward),
        ("ResNet-50 encoder (ImageNet-S50)", test_resnet50_encoder),
        ("Linear classifier", test_linear_classifier),
        ("SupCon loss (single view)", test_supcon_loss),
        ("SupCon loss (distinct classes)", test_supcon_loss_distinct_classes),
        ("Triplet loss (semi-hard)", test_triplet_loss),
        ("Triplet loss (hard)", test_triplet_loss_hard),
        ("CIFAR10 transforms", test_cifar10_transforms),
        ("ImageNet-S50 transforms", test_imagenet_transforms),
        ("TwoCropTransform", test_two_crop_transform),
        ("CE training step (CIFAR10)", test_ce_training_step_cifar10),
        ("SCL training step (CIFAR10)", test_scl_training_step_cifar10),
        ("Triplet training step (CIFAR10)", test_triplet_training_step_cifar10),
        ("CE training step (ImageNet-S50)", test_ce_training_step_imagenet),
        ("SCL training step (ImageNet-S50)", test_scl_training_step_imagenet),
        ("Linear probe training", test_linear_probe_training),
        ("Grad-CAM on ResNet-18", test_gradcam_resnet18),
        ("Grad-CAM on ResNet-50", test_gradcam_resnet50),
        ("EigenCAM on ResNet-18", test_eigencam_resnet18),
        ("Cosine annealing schedule", test_cosine_schedule),
        ("Denormalize utility", test_denormalize),
    ]

    print("=" * 60)
    print(f"CLXAI Test Suite ({len(tests)} tests)")
    print("=" * 60)
    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
