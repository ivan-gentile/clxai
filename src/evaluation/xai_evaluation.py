"""
XAI Evaluation for CIFAR10 models.

Evaluates Grad-CAM and Eigen-CAM explanations using:
- Faithfulness (Pixel Flipping AUC)
- Continuity (SSIM under noise perturbation)
- Contrastivity (SSIM between class-specific explanations, Grad-CAM only)
- Complexity (entropy-based Complexity Metric)
- Sparseness

Usage:
    python -m src.evaluation.xai_evaluation \
        --model_type ce \
        --checkpoint results/cifar10/ce/seed0/best_model.pt \
        --dataset cifar10 --metrics all
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

from src.models.resnet import get_resnet18
from src.utils.data import get_cifar10_loaders


def load_model(model_type, checkpoint_path, classifier_path, device):
    """Load CE or SCL model for CIFAR10."""
    model = get_resnet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "ce":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        encoder_state = checkpoint["model_state_dict"]
        new_state = {}
        for k, v in encoder_state.items():
            if not k.startswith("encoder.") and not k.startswith("fc."):
                new_state[f"encoder.{k}"] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

        if classifier_path and os.path.exists(classifier_path):
            clf_state = torch.load(classifier_path, map_location=device, weights_only=False)
            model.fc.weight.data = clf_state["fc.weight"]
            model.fc.bias.data = clf_state["fc.bias"]

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = True
    return model


def get_cam_methods(model, target_layers):
    return {
        "GradCAM": GradCAM(model=model, target_layers=target_layers),
        "EigenCAM": EigenCAM(model=model, target_layers=target_layers),
    }


def cleanup_cam(cam_obj):
    if hasattr(cam_obj, "activations_and_grads"):
        cam_obj.activations_and_grads.release()


def evaluate_pixel_flipping(model, test_loader, device, num_samples=None):
    """Faithfulness via Pixel Flipping. Lower AUC = more faithful."""
    print("\n[Pixel Flipping] Starting...")
    target_layers = [model.encoder.layer4[-1]]
    pf_metric = quantus.PixelFlipping(perturb_baseline="black", features_in_step=32,
                                       disable_warnings=True, display_progressbar=False)
    cam_methods = get_cam_methods(model, target_layers)
    all_scores, idx = [], 0

    for images, labels in tqdm(test_loader, desc="PF"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cam_methods.values(): cleanup_cam(c)
                return pd.DataFrame(all_scores)
            record = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for name, cam in cam_methods.items():
                with torch.enable_grad():
                    attr = cam(input_tensor=images[i:i+1],
                               targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                if np.all(attr == 0):
                    record[f"{name}_PF_AUC"] = np.nan
                    continue
                pf = pf_metric(model=model, x_batch=images[i:i+1].detach().cpu().numpy(),
                               y_batch=np.array([labels[i].item()]),
                               a_batch=attr[np.newaxis, ...], device=device)[0]
                record[f"{name}_PF_AUC"] = np.trapz(pf) / (len(pf) - 1)
            all_scores.append(record)
            idx += 1
        torch.cuda.empty_cache()

    for c in cam_methods.values(): cleanup_cam(c)
    return pd.DataFrame(all_scores)


def evaluate_continuity(model, test_loader, device, num_samples=None, noise_level=0.02):
    """Continuity via SSIM under Gaussian noise. Higher = better."""
    from skimage.metrics import structural_similarity as ssim
    print("\n[Continuity] Starting...")
    target_layers = [model.encoder.layer4[-1]]
    cam_methods = get_cam_methods(model, target_layers)
    all_scores, idx = [], 0

    for images, labels in tqdm(test_loader, desc="Continuity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cam_methods.values(): cleanup_cam(c)
                return pd.DataFrame(all_scores)
            torch.manual_seed(idx)
            img = images[i:i+1]
            noisy = torch.clamp(img + torch.randn_like(img) * noise_level, 0, 1)
            record = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for name, cam in cam_methods.items():
                with torch.enable_grad():
                    c_orig = cam(input_tensor=img, targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                    c_pert = cam(input_tensor=noisy, targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                record[f"{name}_SSIM_Continuity"] = ssim(c_orig, c_pert, data_range=1.0)
            all_scores.append(record)
            idx += 1
        torch.cuda.empty_cache()

    for c in cam_methods.values(): cleanup_cam(c)
    return pd.DataFrame(all_scores)


def evaluate_contrastivity(model, test_loader, device, num_samples=None):
    """Contrastivity via SSIM between class-specific FAs (Grad-CAM only). Lower = better."""
    from skimage.metrics import structural_similarity as ssim
    print("\n[Contrastivity] Starting...")
    target_layers = [model.encoder.layer4[-1]]
    gradcam = GradCAM(model=model, target_layers=target_layers)
    all_scores, idx = [], 0

    for images, labels in tqdm(test_loader, desc="Contrastivity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
            N, C = logits.shape
            r = torch.randint(0, C - 1, (N,), device=device)
            contrast = r + (r >= preds).long()
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                cleanup_cam(gradcam)
                return pd.DataFrame(all_scores)
            p, c = preds[i].item(), contrast[i].item()
            img = images[i:i+1]
            with torch.enable_grad():
                c_pred = gradcam(input_tensor=img, targets=[ClassifierOutputTarget(p)])[0, :]
                c_other = gradcam(input_tensor=img, targets=[ClassifierOutputTarget(c)])[0, :]
            record = {"idx": idx, "true": labels[i].item(), "pred": p, "contrast_class": c,
                       "GradCAM_SSIM_Contrastivity": ssim(c_pred, c_other, data_range=1.0)}
            all_scores.append(record)
            idx += 1
        torch.cuda.empty_cache()

    cleanup_cam(gradcam)
    return pd.DataFrame(all_scores)


def evaluate_complexity(model, test_loader, device, num_samples=None):
    """Complexity (entropy). Lower = simpler explanation."""
    print("\n[Complexity] Starting...")
    target_layers = [model.encoder.layer4[-1]]
    cm = quantus.Complexity(disable_warnings=True, display_progressbar=False)
    cam_methods = get_cam_methods(model, target_layers)
    all_scores, idx = [], 0

    for images, labels in tqdm(test_loader, desc="Complexity"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cam_methods.values(): cleanup_cam(c)
                return pd.DataFrame(all_scores)
            record = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    attr = cam_obj(input_tensor=images[i:i+1],
                                   targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                if np.all(attr == 0):
                    record[f"{name}_Complexity"] = np.nan
                    continue
                score = cm(model=model, x_batch=images[i:i+1].detach().cpu().numpy(),
                           y_batch=np.array([labels[i].item()]),
                           a_batch=attr[np.newaxis, ...], device=device)[0]
                record[f"{name}_Complexity"] = score
            all_scores.append(record)
            idx += 1
        torch.cuda.empty_cache()

    for c in cam_methods.values(): cleanup_cam(c)
    return pd.DataFrame(all_scores)


def evaluate_sparseness(model, test_loader, device, num_samples=None):
    """Sparseness. Higher = sparser (more focused)."""
    print("\n[Sparseness] Starting...")
    target_layers = [model.encoder.layer4[-1]]
    sp = quantus.Sparseness(disable_warnings=True, display_progressbar=False)
    cam_methods = get_cam_methods(model, target_layers)
    all_scores, idx = [], 0

    for images, labels in tqdm(test_loader, desc="Sparseness"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            if num_samples and idx >= num_samples:
                for c in cam_methods.values(): cleanup_cam(c)
                return pd.DataFrame(all_scores)
            record = {"idx": idx, "true": labels[i].item(), "pred": preds[i].item()}
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    attr = cam_obj(input_tensor=images[i:i+1],
                                   targets=[ClassifierOutputTarget(preds[i].item())])[0, :]
                if np.all(attr == 0):
                    record[f"{name}_Sparseness"] = np.nan
                    continue
                score = sp(model=model, x_batch=images[i:i+1].detach().cpu().numpy(),
                           y_batch=np.array([labels[i].item()]),
                           a_batch=attr[np.newaxis, ...], device=device)[0]
                record[f"{name}_Sparseness"] = score
            all_scores.append(record)
            idx += 1
        torch.cuda.empty_cache()

    for c in cam_methods.values(): cleanup_cam(c)
    return pd.DataFrame(all_scores)


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 XAI Evaluation")
    parser.add_argument("--model_type", required=True, choices=["ce", "scl", "triplet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        choices=["all", "pf", "continuity", "contrastivity", "complexity", "sparseness"])
    parser.add_argument("--output_dir", default="results/xai_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size,
                                         num_workers=4, augment=False)
    model = load_model(args.model_type, args.checkpoint, args.classifier_path, device)

    metrics = args.metrics
    if "all" in metrics:
        metrics = ["pf", "continuity", "contrastivity", "complexity", "sparseness"]

    results = {}
    if "pf" in metrics:
        df = evaluate_pixel_flipping(model, test_loader, device, args.num_samples)
        df.to_csv(output_dir / "pf_scores.csv", index=False)
        results["pf"] = df
    if "continuity" in metrics:
        df = evaluate_continuity(model, test_loader, device, args.num_samples)
        df.to_csv(output_dir / "continuity_scores.csv", index=False)
        results["continuity"] = df
    if "contrastivity" in metrics:
        df = evaluate_contrastivity(model, test_loader, device, args.num_samples)
        df.to_csv(output_dir / "contrastivity_scores.csv", index=False)
        results["contrastivity"] = df
    if "complexity" in metrics:
        df = evaluate_complexity(model, test_loader, device, args.num_samples)
        df.to_csv(output_dir / "complexity_scores.csv", index=False)
        results["complexity"] = df
    if "sparseness" in metrics:
        df = evaluate_sparseness(model, test_loader, device, args.num_samples)
        df.to_csv(output_dir / "sparseness_scores.csv", index=False)
        results["sparseness"] = df

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, df in results.items():
        correct = df[df["true"] == df["pred"]]
        cols = [c for c in df.columns if c not in ["idx", "true", "pred", "contrast_class"]]
        print(f"\n{name.upper()} (n={len(correct)}):")
        for col in cols:
            print(f"  {col}: {correct[col].mean():.4f} ± {correct[col].std():.4f}")


if __name__ == "__main__":
    main()
