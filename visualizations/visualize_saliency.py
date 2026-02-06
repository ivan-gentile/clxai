#!/usr/bin/env python3
"""
Saliency Map Visualization for ImageNet-S50

Loads CE, SCL (SupCon), and TL (Triplet) models, picks random validation images,
generates GradCAM + EigenCAM saliency maps, and produces overlay visualizations
plus a combined grid figure.

Usage:
    python visualizations/visualize_saliency.py --num_images 5 --seed 42
    python visualizations/visualize_saliency.py --num_images 3 --seed 123 --output_dir results/my_saliency
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.imagenet_resnet import get_imagenet_resnet

# ==============================================================================
# Constants
# ==============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default checkpoint paths (relative to project root)
DEFAULT_CHECKPOINTS = {
    "CE": {
        "checkpoint": "results/imagenet50_r50_pretrained/ce_resnet50_imagenet50_best.pt",
        "model_type": "ce",
        "linear_classifier": None,
    },
    "SCL": {
        "checkpoint": "results/imagenet50_r50_experiments/supcon_noaug/best_model.pt",
        "model_type": "encoder",
        "linear_classifier": "results/imagenet50_r50_experiments/supcon_noaug/linear_classifier.pt",
    },
    "TL": {
        "checkpoint": "results/imagenet50_r50_experiments/triplet_noaug/best_model.pt",
        "model_type": "encoder",
        "linear_classifier": "results/imagenet50_r50_experiments/triplet_noaug/linear_classifier.pt",
    },
}

CAM_METHODS = {
    "GradCAM": GradCAM,
    "EigenCAM": EigenCAM,
}

# ==============================================================================
# Model loading (reused from coherence_imagenet50.py)
# ==============================================================================


class EncoderWithClassifier(nn.Module):
    """Wrapper that combines encoder with linear classifier for inference."""

    def __init__(self, encoder, linear_classifier):
        super().__init__()
        self.encoder = encoder
        self.linear_classifier = linear_classifier
        # Expose backbone for Grad-CAM target layer access
        self.backbone = encoder.backbone

    def forward(self, x):
        features = self.encoder.get_embedding(x, normalize=True)
        logits = self.linear_classifier(features)
        return logits


def load_model(checkpoint_path, model_type, linear_classifier_path, device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: 'ce' for classifier, 'encoder' for contrastive encoder
        linear_classifier_path: Path to linear classifier (for encoder only)
        device: torch device

    Returns:
        model: Loaded model
        target_layers: List of target layers for Grad-CAM
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "ce":
        model = get_imagenet_resnet(
            architecture="resnet50_imagenet",
            num_classes=50,
            encoder_only=False,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        # Target layer: model.model is the timm ResNet
        target_layers = [model.model.layer4[-1]]

    elif model_type == "encoder":
        encoder = get_imagenet_resnet(
            architecture="resnet50_imagenet",
            encoder_only=True,
            embedding_dim=128,
            pretrained=False,
        )
        encoder.load_state_dict(checkpoint["model_state_dict"])

        if linear_classifier_path is None:
            raise ValueError("linear_classifier_path required for encoder models")

        linear_state = torch.load(
            linear_classifier_path, map_location=device, weights_only=False
        )
        linear_classifier = nn.Linear(2048, 50)

        # Handle different checkpoint formats
        if "fc.weight" in linear_state:
            linear_classifier.weight.data = linear_state["fc.weight"]
            linear_classifier.bias.data = linear_state["fc.bias"]
        else:
            linear_classifier.load_state_dict(linear_state)

        model = EncoderWithClassifier(encoder, linear_classifier)
        model = model.to(device)
        model.eval()
        # Target layer: encoder.backbone is the timm ResNet
        target_layers = [model.backbone.layer4[-1]]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    return model, target_layers


# ==============================================================================
# Image utilities
# ==============================================================================


def unnormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalization to get an RGB image in [0, 1] float32.

    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std

    Returns:
        RGB image as float32 numpy array [H, W, 3] in [0, 1]
    """
    image = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    image = (image * np.array(std)) + np.array(mean)
    image = image.clip(0, 1)
    return image.astype(np.float32)


def get_validation_transform():
    """Standard ImageNet validation transform (same as used for evaluation)."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ==============================================================================
# Saliency generation
# ==============================================================================


def generate_cam(cam_class, model, target_layers, input_tensor, label, device):
    """
    Generate a single CAM saliency map.

    Args:
        cam_class: CAM class (GradCAM or EigenCAM)
        model: Model for attribution
        target_layers: Target layers for CAM
        input_tensor: Single image tensor [C, H, W]
        label: Integer class label
        device: torch device

    Returns:
        grayscale_cam: Saliency map [H, W] in [0, 1]
    """
    cam = cam_class(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(label)]

    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_batch, targets=targets, aug_smooth=True)[0]

    # Release hooks
    if hasattr(cam, "activations_and_grads"):
        cam.activations_and_grads.release()

    return grayscale_cam


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Saliency Map Visualization for ImageNet-S50"
    )
    parser.add_argument(
        "--num_images", type=int, default=5,
        help="Number of random images to visualize (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for image selection (default: 42)",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "imagenet-s50" / "ImageNetS50"),
        help="Path to ImageNet-S50 data directory",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "results" / "saliency_visualizations"),
        help="Directory to save output images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directories
    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saliency Map Visualization")
    print("=" * 60)
    print(f"Number of images: {args.num_images}")
    print(f"Seed: {args.seed}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Models: {list(DEFAULT_CHECKPOINTS.keys())}")
    print(f"CAM methods: {list(CAM_METHODS.keys())}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 1. Load dataset and pick random images
    # ------------------------------------------------------------------
    print("Loading dataset...")
    dataset = ImageFolder(
        root=os.path.join(args.data_dir, "validation"),
        transform=get_validation_transform(),
    )
    class_names = dataset.classes  # folder names (e.g. 'n01443537')
    # Try to get human-readable class names from folder names
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    num_images = min(args.num_images, len(dataset))
    indices = random.sample(range(len(dataset)), num_images)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Selected indices: {indices}")

    # ------------------------------------------------------------------
    # 2. Load all models
    # ------------------------------------------------------------------
    print("\nLoading models...")
    models = {}
    target_layers = {}

    for model_name, cfg in DEFAULT_CHECKPOINTS.items():
        ckpt_path = PROJECT_ROOT / cfg["checkpoint"]
        linear_path = (
            PROJECT_ROOT / cfg["linear_classifier"]
            if cfg["linear_classifier"]
            else None
        )

        # Validate paths
        if not ckpt_path.exists():
            print(f"  ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        if linear_path and not linear_path.exists():
            print(f"  ERROR: Linear classifier not found: {linear_path}")
            sys.exit(1)

        model, layers = load_model(
            str(ckpt_path), cfg["model_type"], str(linear_path) if linear_path else None, device
        )
        models[model_name] = model
        target_layers[model_name] = layers
        print(f"  {model_name}: loaded ({cfg['model_type']})")

    # ------------------------------------------------------------------
    # 3. Generate saliency maps and save individual images
    # ------------------------------------------------------------------
    print("\nGenerating saliency maps...")

    model_names = list(models.keys())
    cam_names = list(CAM_METHODS.keys())

    # Storage for grid figure: grid[image_idx] = {"original": img, "CE_GradCAM": viz, ...}
    grid_data = []

    for img_i, dataset_idx in enumerate(indices):
        image_tensor, label = dataset[dataset_idx]
        class_name = idx_to_class[label]
        rgb_image = unnormalize_image(image_tensor)

        print(f"\n  Image {img_i + 1}/{num_images}: index={dataset_idx}, "
              f"class={class_name}, label={label}")

        entry = {
            "original": rgb_image,
            "class_name": class_name,
            "dataset_idx": dataset_idx,
            "label": label,
        }

        for model_name in model_names:
            model = models[model_name]
            layers = target_layers[model_name]

            for cam_name, cam_class in CAM_METHODS.items():
                print(f"    {model_name} / {cam_name}...", end=" ", flush=True)

                grayscale_cam = generate_cam(
                    cam_class, model, layers, image_tensor, label, device
                )

                # Overlay heatmap on original image
                viz = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

                key = f"{model_name}_{cam_name}"
                entry[key] = viz

                # Save individual image
                fname = f"img{dataset_idx:04d}_{class_name}_{model_name}_{cam_name}.png"
                fpath = individual_dir / fname
                # show_cam_on_image returns RGB uint8; cv2 expects BGR
                cv2.imwrite(str(fpath), viz[:, :, ::-1])
                print(f"saved {fname}")

        # Also save the original (unnormalized) image
        orig_fname = f"img{dataset_idx:04d}_{class_name}_original.png"
        orig_uint8 = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(individual_dir / orig_fname), orig_uint8[:, :, ::-1])

        grid_data.append(entry)

        # Free GPU memory
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Create combined grid figure
    # ------------------------------------------------------------------
    print("\nCreating grid figure...")

    # Columns: Original | CE GradCAM | CE EigenCAM | SCL GradCAM | SCL EigenCAM | TL GradCAM | TL EigenCAM
    col_keys = ["original"]
    col_labels = ["Original"]
    for mn in model_names:
        for cn in cam_names:
            col_keys.append(f"{mn}_{cn}")
            col_labels.append(f"{mn}\n{cn}")

    n_rows = len(grid_data)
    n_cols = len(col_keys)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for row_i, entry in enumerate(grid_data):
        for col_i, key in enumerate(col_keys):
            ax = axes[row_i, col_i]

            if key == "original":
                img = entry["original"]
            else:
                img = entry[key]

            # Ensure image is displayable (uint8 RGB or float [0,1])
            if img.dtype == np.uint8:
                ax.imshow(img)
            else:
                ax.imshow(img.clip(0, 1))

            ax.set_xticks([])
            ax.set_yticks([])

            # Column header on first row
            if row_i == 0:
                ax.set_title(col_labels[col_i], fontsize=11, fontweight="bold", pad=8)

            # Row label on first column
            if col_i == 0:
                class_label = entry["class_name"]
                ax.set_ylabel(
                    f"{class_label}\n(idx {entry['dataset_idx']})",
                    fontsize=9, rotation=0, labelpad=60, va="center",
                )

    plt.tight_layout()
    grid_path = output_dir / "saliency_grid.png"
    fig.savefig(str(grid_path), dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved grid: {grid_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Individual images: {individual_dir}")
    print(f"  Grid figure: {grid_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
