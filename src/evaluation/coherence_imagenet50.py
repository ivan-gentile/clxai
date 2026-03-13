"""
Coherence Evaluation for ImageNet-S50.

Evaluates XAI coherence metrics (Attribution Localisation, Pointing Game) using
Grad-CAM and Eigen-CAM on ImageNet-S50 segmentation masks.

Usage:
    python -m src.evaluation.coherence_imagenet50 \
        --checkpoint path/to/model.pt --model_type ce \
        --data_dir data/imagenet-s50/ImageNetS50 \
        --output_dir results/coherence/ce
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import quantus

from src.models.imagenet_resnet import get_imagenet_resnet


class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder, linear_classifier):
        super().__init__()
        self.encoder = encoder
        self.linear_classifier = linear_classifier
        self.backbone = encoder.backbone

    def forward(self, x):
        features = self.encoder.get_embedding(x, normalize=True)
        return self.linear_classifier(features)


def create_image_loader(data_dir, batch_size=32, num_workers=8):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def create_mask_loader(data_dir, batch_size=32, num_workers=8):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=os.path.join(data_dir, "validation-segmentation"),
                          transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def load_model(checkpoint_path, model_type, linear_classifier_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "ce":
        model = get_imagenet_resnet(num_classes=50, encoder_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        target_layers = [model.model.layer4[-1]]
    else:
        encoder = get_imagenet_resnet(encoder_only=True, embedding_dim=128)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        linear = nn.Linear(2048, 50)
        lstate = torch.load(linear_classifier_path, map_location=device, weights_only=False)
        if 'fc.weight' in lstate:
            linear.weight.data, linear.bias.data = lstate['fc.weight'], lstate['fc.bias']
        else:
            linear.load_state_dict(lstate)
        model = EncoderWithClassifier(encoder, linear).to(device).eval()
        target_layers = [model.backbone.layer4[-1]]

    for p in model.parameters():
        p.requires_grad = True
    return model, target_layers


def create_saliency_maps(cam_class, model, x_batch, y_batch, target_layers):
    cam = cam_class(model=model, target_layers=target_layers)
    batch_size = x_batch.shape[0]
    attributions = np.zeros((batch_size, 224, 224))
    for i in range(batch_size):
        with torch.enable_grad():
            g = cam(input_tensor=x_batch[i:i+1],
                    targets=[ClassifierOutputTarget(y_batch[i].item())],
                    aug_smooth=True)[0, :]
        if g.max() == 0:
            g[0, 0] += 0.001
        attributions[i] = g
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.release()
    return attributions


def main():
    parser = argparse.ArgumentParser(description="ImageNet-S50 Coherence Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=["ce", "encoder"])
    parser.add_argument("--linear_classifier", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    img_loader = create_image_loader(args.data_dir, args.batch_size, args.num_workers)
    mask_loader = create_mask_loader(args.data_dir, args.batch_size, args.num_workers)
    model, target_layers = load_model(args.checkpoint, args.model_type,
                                       args.linear_classifier, device)

    cam_classes = {"GradCAM": GradCAM, "EigenCAM": EigenCAM}
    attr_loc = quantus.AttributionLocalisation(disable_warnings=True)
    pg = quantus.PointingGame(disable_warnings=True)

    results = {n: {"AL": [], "PG": []} for n in cam_classes}

    for (images, labels), (masks, _) in tqdm(zip(img_loader, mask_loader),
                                               total=len(img_loader), desc="Coherence"):
        images, labels = images.to(device), labels.to(device)
        masks_bin = (masks > 0).long().to(device)

        for cam_name, cam_cls in cam_classes.items():
            attrs = create_saliency_maps(cam_cls, model, images, labels, target_layers)
            x_np, y_np = images.cpu().numpy(), labels.cpu().numpy()
            s_np = masks_bin[:, 0:1, :, :].cpu().numpy()

            try:
                al = attr_loc(model=model, x_batch=x_np, y_batch=y_np,
                              a_batch=attrs, s_batch=s_np, device=device)
                results[cam_name]["AL"].extend(al)
            except Exception:
                pass
            try:
                p = pg(model=model, x_batch=x_np, y_batch=y_np,
                       a_batch=attrs, s_batch=s_np, device=device)
                results[cam_name]["PG"].extend(p)
            except Exception:
                pass
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\nCoherence Results\n{'='*60}")
    for cam_name, metrics in results.items():
        print(f"\n{cam_name}:")
        for metric_name, scores in metrics.items():
            clean = [s for s in scores if not np.isnan(s)]
            if clean:
                print(f"  {metric_name}: {np.mean(clean):.4f} ± {np.std(clean):.4f}")
                pd.DataFrame({metric_name: clean}).to_csv(
                    out / f"{cam_name}_{metric_name}_scores.csv", index=False)


if __name__ == "__main__":
    main()
