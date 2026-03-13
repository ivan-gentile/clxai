#!/usr/bin/env python3
"""
Saliency Map Visualization for ImageNet-S50.

Loads CE, SCL, and TL models, generates Grad-CAM and Eigen-CAM saliency maps,
and produces overlay visualizations plus a combined grid figure.

Usage:
    python scripts/visualize_saliency.py \
        --data_dir data/imagenet-s50/ImageNetS50 \
        --num_images 5 --seed 42
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.imagenet_resnet import get_imagenet_resnet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CAM_METHODS = {"GradCAM": GradCAM, "EigenCAM": EigenCAM}


class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder, linear_classifier):
        super().__init__()
        self.encoder = encoder
        self.linear_classifier = linear_classifier
        self.backbone = encoder.backbone

    def forward(self, x):
        features = self.encoder.get_embedding(x, normalize=True)
        return self.linear_classifier(features)


def load_model(checkpoint_path, model_type, linear_classifier_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "ce":
        model = get_imagenet_resnet(num_classes=50, encoder_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        target_layers = [model.model.layer4[-1]]
    else:
        encoder = get_imagenet_resnet(encoder_only=True, embedding_dim=128)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        linear = nn.Linear(2048, 50)
        lstate = torch.load(linear_classifier_path, map_location=device, weights_only=False)
        if "fc.weight" in lstate:
            linear.weight.data, linear.bias.data = lstate["fc.weight"], lstate["fc.bias"]
        else:
            linear.load_state_dict(lstate)
        model = EncoderWithClassifier(encoder, linear).to(device).eval()
        target_layers = [model.backbone.layer4[-1]]

    for p in model.parameters():
        p.requires_grad = True
    return model, target_layers


def unnormalize_image(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    return image.clip(0, 1).astype(np.float32)


def get_validation_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def generate_cam(cam_class, model, target_layers, input_tensor, label, device):
    cam = cam_class(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(label)]
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_batch, targets=targets, aug_smooth=True)[0]
    if hasattr(cam, "activations_and_grads"):
        cam.activations_and_grads.release()
    return grayscale_cam


def main():
    parser = argparse.ArgumentParser(description="Saliency Map Visualization")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet-S50 data (with validation/ subfolder)")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Checkpoint paths in format NAME:TYPE:PATH[:LINEAR_PATH]")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/saliency_visualizations")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(root=os.path.join(args.data_dir, "validation"),
                          transform=get_validation_transform())
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    num_images = min(args.num_images, len(dataset))
    indices = random.sample(range(len(dataset)), num_images)

    models, target_layers_dict = {}, {}
    for spec in args.checkpoints:
        parts = spec.split(":")
        name, mtype, path = parts[0], parts[1], parts[2]
        linear = parts[3] if len(parts) > 3 else None
        model, tl = load_model(path, mtype, linear, device)
        models[name], target_layers_dict[name] = model, tl
        print(f"Loaded {name} ({mtype})")

    model_names = list(models.keys())
    cam_names = list(CAM_METHODS.keys())
    grid_data = []

    for img_i, dataset_idx in enumerate(indices):
        image_tensor, label = dataset[dataset_idx]
        class_name = idx_to_class[label]
        rgb_image = unnormalize_image(image_tensor)
        print(f"\nImage {img_i+1}/{num_images}: idx={dataset_idx}, class={class_name}")

        entry = {"original": rgb_image, "class_name": class_name, "dataset_idx": dataset_idx}
        for mn in model_names:
            for cn, cam_cls in CAM_METHODS.items():
                gc = generate_cam(cam_cls, models[mn], target_layers_dict[mn],
                                  image_tensor, label, device)
                viz = show_cam_on_image(rgb_image, gc, use_rgb=True)
                entry[f"{mn}_{cn}"] = viz
                fname = f"img{dataset_idx:04d}_{class_name}_{mn}_{cn}.png"
                cv2.imwrite(str(individual_dir / fname), viz[:, :, ::-1])

        orig_uint8 = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(individual_dir / f"img{dataset_idx:04d}_{class_name}_original.png"),
                    orig_uint8[:, :, ::-1])
        grid_data.append(entry)
        torch.cuda.empty_cache()

    col_keys = ["original"] + [f"{mn}_{cn}" for mn in model_names for cn in cam_names]
    col_labels = ["Original"] + [f"{mn}\n{cn}" for mn in model_names for cn in cam_names]
    n_rows, n_cols = len(grid_data), len(col_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 3.2*n_rows), squeeze=False)
    for row_i, entry in enumerate(grid_data):
        for col_i, key in enumerate(col_keys):
            ax = axes[row_i, col_i]
            img = entry.get(key, entry["original"])
            ax.imshow(img if img.dtype == np.uint8 else img.clip(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(col_labels[col_i], fontsize=11, fontweight="bold", pad=8)
            if col_i == 0:
                ax.set_ylabel(f"{entry['class_name']}\n(idx {entry['dataset_idx']})",
                              fontsize=9, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    grid_path = output_dir / "saliency_grid.png"
    fig.savefig(str(grid_path), dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"\nGrid saved: {grid_path}")


if __name__ == "__main__":
    main()
