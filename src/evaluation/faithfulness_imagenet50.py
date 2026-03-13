"""
XAI Evaluation for ImageNet-S50 models (faithfulness, continuity, contrastivity, complexity).

Usage:
    python -m src.evaluation.faithfulness_imagenet50 \
        --checkpoint path/to/best_model.pt \
        --model_type ce --data_dir path/to/imagenet-1k --metrics all
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import quantus

from src.models.imagenet_resnet import get_imagenet_resnet
from src.utils.imagenet_data import get_imagenet50_loaders


class EncoderWithClassifier(nn.Module):
    """Wrapper combining contrastive encoder with linear classifier."""
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


def get_cam_methods(model, target_layers):
    return {
        "GradCAM": GradCAM(model=model, target_layers=target_layers),
        "EigenCAM": EigenCAM(model=model, target_layers=target_layers),
    }


def cleanup_cam(cam_obj):
    if hasattr(cam_obj, "activations_and_grads"):
        cam_obj.activations_and_grads.release()


def evaluate_pixel_flipping(model, loader, target_layers, device, num_samples=None):
    print("\n[Pixel Flipping] Starting...")
    pf = quantus.PixelFlipping(perturb_baseline="black", features_in_step=224,
                                disable_warnings=True, display_progressbar=False)
    cams = get_cam_methods(model, target_layers)
    scores, idx = [], 0
    for images, labels in tqdm(loader, desc="PF"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cams.values(): cleanup_cam(c)
                return pd.DataFrame(scores)
            rec = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for n, cam in cams.items():
                with torch.enable_grad():
                    a = cam(input_tensor=images[i:i+1], targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                if np.all(a == 0): rec[f"{n}_PF_AUC"] = np.nan; continue
                s = pf(model=model, x_batch=images[i:i+1].detach().cpu().numpy(),
                       y_batch=np.array([labels[i].item()]), a_batch=a[np.newaxis, ...], device=device)[0]
                rec[f"{n}_PF_AUC"] = np.trapz(s) / (len(s) - 1)
            scores.append(rec); idx += 1
        torch.cuda.empty_cache()
    for c in cams.values(): cleanup_cam(c)
    return pd.DataFrame(scores)


def evaluate_continuity(model, loader, target_layers, device, num_samples=None, noise=0.02):
    from skimage.metrics import structural_similarity as ssim
    print("\n[Continuity] Starting...")
    cams = get_cam_methods(model, target_layers)
    scores, idx = [], 0
    for images, labels in tqdm(loader, desc="Continuity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cams.values(): cleanup_cam(c)
                return pd.DataFrame(scores)
            torch.manual_seed(idx)
            img, noisy = images[i:i+1], torch.clamp(images[i:i+1] + torch.randn_like(images[i:i+1]) * noise, 0, 1)
            rec = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for n, cam in cams.items():
                with torch.enable_grad():
                    co = cam(input_tensor=img, targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                    cp = cam(input_tensor=noisy, targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                rec[f"{n}_SSIM_Continuity"] = ssim(co, cp, data_range=1.0)
            scores.append(rec); idx += 1
        torch.cuda.empty_cache()
    for c in cams.values(): cleanup_cam(c)
    return pd.DataFrame(scores)


def evaluate_contrastivity(model, loader, target_layers, device, num_samples=None):
    from skimage.metrics import structural_similarity as ssim
    print("\n[Contrastivity] Starting...")
    gc = GradCAM(model=model, target_layers=target_layers)
    scores, idx = [], 0
    for images, labels in tqdm(loader, desc="Contrastivity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images); preds = logits.argmax(dim=1)
            N, C = logits.shape
            r = torch.randint(0, C-1, (N,), device=device)
            contrast = r + (r >= preds).long()
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                cleanup_cam(gc); return pd.DataFrame(scores)
            p, c = preds[i].item(), contrast[i].item()
            with torch.enable_grad():
                a1 = gc(input_tensor=images[i:i+1], targets=[ClassifierOutputTarget(p)])[0, :]
                a2 = gc(input_tensor=images[i:i+1], targets=[ClassifierOutputTarget(c)])[0, :]
            scores.append({"idx": idx, "true": labels[i].item(), "pred": p,
                           "GradCAM_SSIM_Contrastivity": ssim(a1, a2, data_range=1.0)})
            idx += 1
        torch.cuda.empty_cache()
    cleanup_cam(gc)
    return pd.DataFrame(scores)


def evaluate_complexity(model, loader, target_layers, device, num_samples=None):
    print("\n[Complexity] Starting...")
    cm = quantus.Complexity(disable_warnings=True, display_progressbar=False)
    cams = get_cam_methods(model, target_layers)
    scores, idx = [], 0
    for images, labels in tqdm(loader, desc="Complexity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cams.values(): cleanup_cam(c)
                return pd.DataFrame(scores)
            rec = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for n, cam in cams.items():
                with torch.enable_grad():
                    a = cam(input_tensor=images[i:i+1], targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                if np.all(a == 0): rec[f"{n}_Complexity"] = np.nan; continue
                s = cm(model=model, x_batch=images[i:i+1].detach().cpu().numpy(),
                       y_batch=np.array([labels[i].item()]), a_batch=a[np.newaxis, ...], device=device)[0]
                rec[f"{n}_Complexity"] = s
            scores.append(rec); idx += 1
        torch.cuda.empty_cache()
    for c in cams.values(): cleanup_cam(c)
    return pd.DataFrame(scores)


def main():
    parser = argparse.ArgumentParser(description="ImageNet-S50 XAI Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=["ce", "encoder"])
    parser.add_argument("--linear_classifier", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        choices=["all", "pf", "continuity", "contrastivity", "complexity"])
    parser.add_argument("--output_dir", default="results/evals-imagenet50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = args.model_name or "model"
    out = Path(args.output_dir) / name; out.mkdir(parents=True, exist_ok=True)

    _, loader = get_imagenet50_loaders(data_dir=args.data_dir, batch_size=args.batch_size,
                                        num_workers=8, augment=False, contrastive=False)
    model, tl = load_model(args.checkpoint, args.model_type, args.linear_classifier, device)

    metrics = args.metrics
    if "all" in metrics: metrics = ["pf", "continuity", "contrastivity", "complexity"]

    for m in metrics:
        if m == "pf":
            df = evaluate_pixel_flipping(model, loader, tl, device, args.num_samples)
        elif m == "continuity":
            df = evaluate_continuity(model, loader, tl, device, args.num_samples)
        elif m == "contrastivity":
            df = evaluate_contrastivity(model, loader, tl, device, args.num_samples)
        elif m == "complexity":
            df = evaluate_complexity(model, loader, tl, device, args.num_samples)
        df.to_csv(out / f"{m}_scores.csv", index=False)
        print(f"  Saved: {m}_scores.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
