#!/usr/bin/env python3
"""
CLXAI - XAI Faithfulness Evaluation Script

Evaluates XAI methods (GradCAM, EigenCAM, AblationCAM) using multiple
faithfulness metrics on CE vs SCL trained models.

Usage:
    python scripts/run_xai_evaluation.py --model_version ce --metrics pf irof
    python scripts/run_xai_evaluation.py --model_version scl --metrics all
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_grad_cam import GradCAM, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import quantus

from src.models.resnet import get_model
from src.utils.data import get_data_loaders, get_num_classes

# Import custom IROF
from my_irof import IROF


def parse_args():
    parser = argparse.ArgumentParser(description="CLXAI XAI Evaluation")
    parser.add_argument("--model_version", type=str, default="ce",
                        choices=["ce", "scl", "triplet"], help="Model type to evaluate")
    parser.add_argument("--augmentation", type=str, default="pixel",
                        choices=["pixel", "pixel50"], help="Augmentation type used in training")
    parser.add_argument("--model_seed", type=int, default=0,
                        help="Model seed (0-4)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (overrides auto-detection)")
    parser.add_argument("--classifier_path", type=str, default=None,
                        help="Path to linear classifier (for SCL/triplet)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory containing datasets")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for data loading")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None = all)")
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        choices=["all", "pf", "irof", "sparseness", 
                                 "complexity", "robustness", "contrastivity"],
                        help="Metrics to compute")
    parser.add_argument("--output_dir", type=str, default="results/xai_eval",
                        help="Directory to save results")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save visualization plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def setup_device():
    """Setup CUDA device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_model(model_version, model_path, classifier_path, dataset, device):
    """
    Load CE or SCL model.
    
    Args:
        model_version: 'ce', 'scl', or 'triplet'
        model_path: Path to best_model.pt checkpoint
        classifier_path: Path to linear_classifier.pt (for SCL/triplet only)
        dataset: Dataset name for num_classes
        device: torch device
    """
    architecture = "resnet18"
    num_classes = get_num_classes(dataset)
    
    model = get_model(architecture=architecture, num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if model_version == "ce":
        # CE checkpoint has 'encoder.xxx' keys - load directly
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"  Loaded CE model (test_acc: {checkpoint.get('test_acc', 'N/A')})")
    
    elif model_version in ["scl", "triplet"]:
        # SCL/Triplet checkpoint has 'xxx' keys (encoder only, no prefix)
        # Need to add 'encoder.' prefix and load linear classifier separately
        encoder_state_dict = checkpoint["model_state_dict"]
        
        # Add 'encoder.' prefix to encoder weights
        new_state_dict = {}
        for k, v in encoder_state_dict.items():
            if not k.startswith("encoder.") and not k.startswith("fc."):
                new_state_dict[f"encoder.{k}"] = v
            else:
                new_state_dict[k] = v
        
        # Load encoder weights
        model.load_state_dict(new_state_dict, strict=False)
        
        # Load linear classifier
        if classifier_path and os.path.exists(classifier_path):
            classifier_state = torch.load(classifier_path, map_location=device, weights_only=False)
            # classifier has 'fc.weight' and 'fc.bias'
            model.fc.weight.data = classifier_state["fc.weight"]
            model.fc.bias.data = classifier_state["fc.bias"]
            print(f"  Loaded {model_version.upper()} encoder + linear classifier")
        else:
            print(f"  WARNING: No linear classifier found at {classifier_path}")
            print(f"  Loaded {model_version.upper()} encoder only (knn_acc: {checkpoint.get('knn_acc', 'N/A')})")
    
    model.to(device)
    model.eval()
    
    # Ensure gradients are enabled for XAI methods
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def get_cam_methods(model, target_layers):
    """Initialize CAM methods."""
    return {
        "GradCAM": GradCAM(model=model, target_layers=target_layers),
        "EigenCAM": EigenCAM(model=model, target_layers=target_layers),
        "AblationCAM": AblationCAM(model=model, target_layers=target_layers)
    }


def cleanup_cam(cam_obj):
    """Release CAM resources."""
    if hasattr(cam_obj, "activations_and_grads"):
        cam_obj.activations_and_grads.release()


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_pixel_flipping(model, test_loader, device, num_samples=None, save_plots=False, output_dir=None):
    """
    Evaluate Pixel Flipping metric.
    Lower AUC = more faithful explanation.
    """
    print("\n[Pixel Flipping] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    pixel_flipping_metric = quantus.PixelFlipping(
        perturb_baseline="black",
        features_in_step=32,
        disable_warnings=True,
    )
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Pixel Flipping")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            p_idx = preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            record = {"idx": global_idx, "true": g_idx, "pred": p_idx}
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    grayscale_cam = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                
                pf_score = pixel_flipping_metric(
                    model=model,
                    x_batch=img_tensor.detach().cpu().numpy(),
                    y_batch=np.array([g_idx]),
                    a_batch=grayscale_cam[np.newaxis, ...],
                    device=device
                )[0]
                
                auc_val = np.trapz(pf_score) / (len(pf_score) - 1)
                record[f"{name}_PF_AUC"] = auc_val
                
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


def evaluate_irof(model, test_loader, device, num_samples=None, 
                  segmentation_method="slic", perturb_baseline="black",
                  save_plots=False, output_dir=None):
    """
    Evaluate IROF (Iterative Removal of Features) metric.
    Higher AOC = more faithful explanation.
    """
    print("\n[IROF] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    irof_metric = IROF(
        segmentation_method=segmentation_method,
        perturb_baseline=perturb_baseline,
        abs=False,
        normalise=True,
        disable_warnings=True,
        return_scores=True,
        distance_based=False,
        return_aggregate=False,
    )
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="IROF")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            p_idx = preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            record = {"idx": global_idx, "true": g_idx, "pred": p_idx}
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    grayscale_cam = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                
                x_np = img_tensor.detach().cpu().numpy()
                a_np = np.repeat(grayscale_cam[np.newaxis, :, :], x_np.shape[1], axis=0)
                a_np = a_np[np.newaxis, ...]
                
                try:
                    irof_result = irof_metric(
                        model=model,
                        x_batch=x_np,
                        y_batch=np.array([g_idx]),
                        a_batch=a_np,
                        device=device
                    )
                    
                    if isinstance(irof_result, list) and len(irof_result) > 0:
                        result_item = irof_result[0]
                        if isinstance(result_item, tuple) and len(result_item) == 2:
                            aoc_score, irof_curve = result_item
                            record[f"{name}_IROF_AOC"] = float(aoc_score)
                        else:
                            record[f"{name}_IROF_AOC"] = float(result_item)
                    else:
                        record[f"{name}_IROF_AOC"] = np.nan
                        
                except Exception as e:
                    print(f"Error IROF for {name}, sample {global_idx}: {e}")
                    record[f"{name}_IROF_AOC"] = np.nan
                
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


def evaluate_sparseness(model, test_loader, device, num_samples=None, 
                        save_plots=False, output_dir=None):
    """
    Evaluate Sparseness metric.
    Higher = sparser (more focused) explanation.
    """
    print("\n[Sparseness] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    sparsity_metric = quantus.Sparseness(disable_warnings=True)
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Sparseness")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            p_idx = preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            record = {"idx": global_idx, "true": g_idx, "pred": p_idx}
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    grayscale_cam = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                
                sparsity_score = sparsity_metric(
                    model=model,
                    x_batch=img_tensor.detach().cpu().numpy(),
                    y_batch=np.array([g_idx]),
                    a_batch=grayscale_cam[np.newaxis, ...],
                    device=device
                )[0]
                
                record[f"{name}_Sparsity"] = sparsity_score
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


def evaluate_complexity(model, test_loader, device, num_samples=None,
                        save_plots=False, output_dir=None):
    """
    Evaluate Complexity metric.
    Lower = simpler (more interpretable) explanation.
    """
    print("\n[Complexity] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    complexity_metric = quantus.Complexity(disable_warnings=True)
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Complexity")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            p_idx = preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            record = {"idx": global_idx, "true": g_idx, "pred": p_idx}
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    grayscale_cam = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                
                complexity_score = complexity_metric(
                    model=model,
                    x_batch=img_tensor.detach().cpu().numpy(),
                    y_batch=np.array([g_idx]),
                    a_batch=grayscale_cam[np.newaxis, ...],
                    device=device
                )[0]
                
                record[f"{name}_Complexity"] = complexity_score
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


def evaluate_robustness(model, test_loader, device, num_samples=None,
                        noise_level=0.02, save_plots=False, output_dir=None):
    """
    Evaluate SSIM Robustness metric.
    Higher SSIM = more robust explanation (stable under noise).
    """
    from skimage.metrics import structural_similarity as ssim
    
    print("\n[Robustness] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Robustness")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            torch.manual_seed(global_idx)
            np.random.seed(global_idx)
            
            p_idx = preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            # Add noise
            perturbed_tensor = img_tensor + torch.randn_like(img_tensor) * noise_level
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
            
            record = {"idx": global_idx, "true": g_idx, "pred": p_idx}
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    cam_orig = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                    
                    cam_pert = cam_obj(
                        input_tensor=perturbed_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                
                robustness_score = ssim(cam_orig, cam_pert, data_range=1.0)
                record[f"{name}_SSIM_Robustness"] = robustness_score
                
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


def evaluate_contrastivity(model, test_loader, device, num_samples=None,
                           save_plots=False, output_dir=None):
    """
    Evaluate SSIM Contrastivity metric.
    Compares explanations for predicted class vs a random different class.
    Lower SSIM = explanations are more contrastive (class-specific).
    """
    from skimage.metrics import structural_similarity as ssim
    
    print("\n[Contrastivity] Starting evaluation...")
    model.eval()
    target_layers = [model.encoder.layer4[-1]]
    
    all_scores = []
    global_idx = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Contrastivity")):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            N, C = logits.shape
            r = torch.randint(0, C - 1, (N,), device=device)
            contrast_preds = r + (r >= preds).long()
        
        for i in range(images.size(0)):
            if num_samples and global_idx >= num_samples:
                return pd.DataFrame(all_scores)
            
            p_idx = preds[i].item()
            c_idx = contrast_preds[i].item()
            g_idx = labels[i].item()
            img_tensor = images[i:i+1]
            
            record = {
                "idx": global_idx, 
                "true": g_idx, 
                "pred": p_idx,
                "contrast_class": c_idx
            }
            
            cam_methods = get_cam_methods(model, target_layers)
            
            for name, cam_obj in cam_methods.items():
                with torch.enable_grad():
                    cam_factual = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(p_idx)]
                    )[0, :]
                    
                    cam_contrastive = cam_obj(
                        input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(c_idx)]
                    )[0, :]
                
                contrast_ssim = ssim(cam_factual, cam_contrastive, data_range=1.0)
                contrast_l2 = np.linalg.norm(cam_factual - cam_contrastive)
                
                record[f"{name}_SSIM_Contrastivity"] = contrast_ssim
                record[f"{name}_L2_Contrastivity"] = contrast_l2
                
                cleanup_cam(cam_obj)
            
            all_scores.append(record)
            global_idx += 1
            del cam_methods
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_scores)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device()
    
    # Output directory - include model info in path
    output_dir = Path(args.output_dir) / f"{args.model_version}_{args.augmentation}_seed{args.model_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect model paths if not provided
    if args.model_path is None:
        model_dir = PROJECT_ROOT / "export_best_models" / args.dataset / f"{args.model_version}_{args.augmentation}" / f"seed{args.model_seed}"
        args.model_path = str(model_dir / "best_model.pt")
        
        # For SCL/triplet, also set classifier path
        if args.model_version in ["scl", "triplet"] and args.classifier_path is None:
            args.classifier_path = str(model_dir / "linear_classifier.pt")
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print(f"Available models:")
        models_dir = PROJECT_ROOT / "export_best_models" / args.dataset
        if models_dir.exists():
            for d in models_dir.iterdir():
                print(f"  - {d.name}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"CLXAI XAI Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model_version}_{args.augmentation} (seed {args.model_seed})")
    print(f"Model path: {args.model_path}")
    if args.classifier_path:
        print(f"Classifier: {args.classifier_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Data dir: {args.data_dir}")
    print(f"Metrics: {args.metrics}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    _, test_loader = get_data_loaders(
        dataset=args.dataset,
        data_dir=str(PROJECT_ROOT / args.data_dir),
        batch_size=args.batch_size,
        num_workers=4,
        augment=False,
    )
    
    # Load model
    print("Loading model...")
    model = load_model(
        model_version=args.model_version,
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        dataset=args.dataset,
        device=device
    )
    
    # Determine which metrics to run
    metrics_to_run = args.metrics
    if "all" in metrics_to_run:
        metrics_to_run = ["pf", "irof", "sparseness", "complexity", "robustness", "contrastivity"]
    
    # Run evaluations
    results = {}
    
    if "pf" in metrics_to_run:
        df = evaluate_pixel_flipping(
            model, test_loader, device, 
            num_samples=args.num_samples,
            save_plots=args.save_plots, 
            output_dir=output_dir
        )
        results["pf"] = df
        df.to_csv(output_dir / "pf_scores.csv", index=False)
        print(f"  Saved: pf_scores.csv")
    
    if "irof" in metrics_to_run:
        df = evaluate_irof(
            model, test_loader, device,
            num_samples=args.num_samples,
            save_plots=args.save_plots,
            output_dir=output_dir
        )
        results["irof"] = df
        df.to_csv(output_dir / "irof_scores.csv", index=False)
        print(f"  Saved: irof_scores.csv")
    
    if "sparseness" in metrics_to_run:
        df = evaluate_sparseness(
            model, test_loader, device,
            num_samples=args.num_samples,
            save_plots=args.save_plots,
            output_dir=output_dir
        )
        results["sparseness"] = df
        df.to_csv(output_dir / "sparseness_scores.csv", index=False)
        print(f"  Saved: sparseness_scores.csv")
    
    if "complexity" in metrics_to_run:
        df = evaluate_complexity(
            model, test_loader, device,
            num_samples=args.num_samples,
            save_plots=args.save_plots,
            output_dir=output_dir
        )
        results["complexity"] = df
        df.to_csv(output_dir / "complexity_scores.csv", index=False)
        print(f"  Saved: complexity_scores.csv")
    
    if "robustness" in metrics_to_run:
        df = evaluate_robustness(
            model, test_loader, device,
            num_samples=args.num_samples,
            save_plots=args.save_plots,
            output_dir=output_dir
        )
        results["robustness"] = df
        df.to_csv(output_dir / "robustness_scores.csv", index=False)
        print(f"  Saved: robustness_scores.csv")
    
    if "contrastivity" in metrics_to_run:
        df = evaluate_contrastivity(
            model, test_loader, device,
            num_samples=args.num_samples,
            save_plots=args.save_plots,
            output_dir=output_dir
        )
        results["contrastivity"] = df
        df.to_csv(output_dir / "contrastivity_scores.csv", index=False)
        print(f"  Saved: contrastivity_scores.csv")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for metric_name, df in results.items():
        print(f"\n{metric_name.upper()}:")
        # Filter to correct predictions only
        correct_mask = df["true"] == df["pred"]
        df_correct = df[correct_mask]
        print(f"  Accuracy: {correct_mask.mean()*100:.2f}%")
        
        # Print metric-specific columns
        metric_cols = [c for c in df.columns if c not in ["idx", "true", "pred", "contrast_class"]]
        for col in metric_cols:
            print(f"  {col}: {df_correct[col].mean():.4f} (+/- {df_correct[col].std():.4f})")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
