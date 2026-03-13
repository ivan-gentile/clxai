"""
Test XAI evaluation pipeline components (lightweight for login nodes).
Heavy tests (quantus metrics) require GPU nodes.
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


def test_ssim_computation():
    """SSIM computation between two attribution maps."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    a1 = np.random.rand(32, 32).astype(np.float32)
    a2 = a1 + np.random.randn(32, 32).astype(np.float32) * 0.01
    val = ssim(a1, a2, data_range=1.0)
    assert isinstance(val, float)
    assert -1 <= val <= 1
    assert val > 0.9, "Slight perturbation should give high SSIM"


def test_quantus_imports():
    """Verify quantus metrics are importable."""
    import quantus
    assert hasattr(quantus, 'PixelFlipping')
    assert hasattr(quantus, 'Complexity')
    assert hasattr(quantus, 'Sparseness')
    pf = quantus.PixelFlipping(perturb_baseline="black", features_in_step=32,
                                disable_warnings=True, display_progressbar=False)
    cm = quantus.Complexity(disable_warnings=True, display_progressbar=False)
    sp = quantus.Sparseness(disable_warnings=True, display_progressbar=False)
    assert pf is not None and cm is not None and sp is not None


def test_xai_eval_function_signatures():
    """Verify evaluation functions have expected signatures."""
    import inspect
    from src.evaluation.xai_evaluation import (
        evaluate_pixel_flipping, evaluate_continuity,
        evaluate_contrastivity, evaluate_complexity, evaluate_sparseness
    )
    for fn in [evaluate_pixel_flipping, evaluate_continuity,
               evaluate_contrastivity, evaluate_complexity, evaluate_sparseness]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert 'model' in params, f"{fn.__name__} missing 'model' param"
        assert 'device' in params, f"{fn.__name__} missing 'device' param"


def test_imagenet_eval_function_signatures():
    """Verify ImageNet evaluation functions have expected signatures."""
    import inspect
    from src.evaluation.faithfulness_imagenet50 import (
        evaluate_pixel_flipping, evaluate_continuity,
        evaluate_contrastivity, evaluate_complexity
    )
    for fn in [evaluate_pixel_flipping, evaluate_continuity,
               evaluate_contrastivity, evaluate_complexity]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert 'model' in params
        assert 'target_layers' in params


def test_coherence_eval_function_signatures():
    """Verify coherence evaluation functions exist and have expected signatures."""
    import inspect
    from src.evaluation.coherence_imagenet50 import (
        create_image_loader, create_mask_loader, create_saliency_maps, load_model
    )
    sig = inspect.signature(create_saliency_maps)
    params = list(sig.parameters.keys())
    assert 'cam_class' in params
    assert 'model' in params
    assert 'target_layers' in params


def test_encoder_with_classifier_wrapper():
    """Verify the encoder+classifier pattern works for both CE and CL models."""
    import torch
    import torch.nn as nn
    from src.models.imagenet_resnet import get_imagenet_resnet

    ce_model = get_imagenet_resnet(num_classes=50, encoder_only=False)
    assert hasattr(ce_model, 'model')
    assert hasattr(ce_model.model, 'layer4')

    encoder = get_imagenet_resnet(encoder_only=True, embedding_dim=128)
    assert hasattr(encoder, 'backbone')
    assert hasattr(encoder.backbone, 'layer4')

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ce_out = ce_model(x)
        assert ce_out.shape == (1, 50)
        enc_out = encoder(x)
        assert enc_out.shape == (1, 128)
        raw = encoder.get_embedding(x, normalize=False)
        assert raw.shape == (1, 2048)


def test_cam_target_layer_consistency():
    """Verify Grad-CAM target layers match between training and evaluation code."""
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.models.resnet import get_resnet18

    model = get_resnet18(num_classes=10)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

    target_layers = [model.encoder.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    x = torch.randn(1, 3, 32, 32)
    with torch.enable_grad():
        attr = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])
    assert attr.shape == (1, 32, 32)
    assert attr.max() > 0

    attr2 = cam(input_tensor=x, targets=[ClassifierOutputTarget(5)])
    assert attr2.shape == (1, 32, 32)
    cam.activations_and_grads.release()


def test_training_script_imports():
    """Verify all training scripts import without errors."""
    import importlib
    for name in ['src.training.train_ce', 'src.training.train_scl',
                 'src.training.train_ce_imagenet50', 'src.training.train_scl_imagenet50']:
        mod = importlib.import_module(name)
        assert hasattr(mod, 'main'), f"{name} missing main()"


def test_config_files_parseable():
    """Verify all YAML config files are parseable."""
    import yaml
    from pathlib import Path
    config_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'configs'
    expected_configs = ['ce_cifar10', 'scl_cifar10', 'triplet_cifar10',
                        'ce_imagenet50', 'scl_imagenet50', 'triplet_imagenet50']
    for name in expected_configs:
        path = config_dir / f'{name}.yaml'
        assert path.exists(), f"Config {name}.yaml not found"
        with open(path) as f:
            config = yaml.safe_load(f)
        assert 'model' in config, f"{name}.yaml missing 'model' section"
        assert 'training' in config, f"{name}.yaml missing 'training' section"
        assert 'data' in config, f"{name}.yaml missing 'data' section"
        assert 'output' in config, f"{name}.yaml missing 'output' section"


if __name__ == "__main__":
    tests = [
        ("SSIM computation", test_ssim_computation),
        ("Quantus imports", test_quantus_imports),
        ("XAI eval function signatures", test_xai_eval_function_signatures),
        ("ImageNet eval function signatures", test_imagenet_eval_function_signatures),
        ("Coherence eval function signatures", test_coherence_eval_function_signatures),
        ("Encoder+Classifier wrapper", test_encoder_with_classifier_wrapper),
        ("CAM target layer consistency", test_cam_target_layer_consistency),
        ("Training script imports", test_training_script_imports),
        ("Config files parseable", test_config_files_parseable),
    ]

    print("=" * 60)
    print(f"CLXAI XAI Pipeline Tests ({len(tests)} tests)")
    print("=" * 60)
    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
