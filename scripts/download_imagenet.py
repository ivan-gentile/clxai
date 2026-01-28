#!/usr/bin/env python3
"""
Download ImageNet-1k dataset from Hugging Face to the data directory.

IMPORTANT: Before running this script, you must:
1. Create a Hugging Face account at https://huggingface.co
2. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Accept the ImageNet terms of access
4. Create an access token at https://huggingface.co/settings/tokens

Run this script via SLURM job (use download_imagenet.sh) as it requires:
- Internet access (only available on login nodes, but download is too large)
- Significant disk space (~155GB)
- Long runtime

Usage:
    python download_imagenet.py [--streaming] [--subset SUBSET]

Options:
    --streaming     Use streaming mode (doesn't download full dataset)
    --subset        Download specific subset: 'train', 'validation', or 'test'
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Hugging Face authentication token
# This token must have access to gated datasets (ImageNet requires accepting terms)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Set environment variables for Hugging Face
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Data directories
DATA_DIR = "/leonardo_scratch/fast/CNHPC_1905882/clxai/data"
IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet-1k")

# Hugging Face cache directories (use scratch for speed)
HF_CACHE_DIR = "/leonardo_scratch/fast/CNHPC_1905882/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")


def setup_authentication():
    """Setup Hugging Face authentication."""
    from huggingface_hub import login, HfApi
    
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("Successfully logged in to Hugging Face")
        
        # Verify access to the dataset
        api = HfApi(token=HF_TOKEN)
        try:
            api.dataset_info("ILSVRC/imagenet-1k")
            logger.info("Verified access to ILSVRC/imagenet-1k dataset")
        except Exception as e:
            logger.error(f"Cannot access ImageNet dataset: {e}")
            logger.error("Please ensure you have accepted the terms at:")
            logger.error("https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)


def download_imagenet_hf(subset=None, streaming=False):
    """
    Download ImageNet-1k using Hugging Face datasets library.
    
    Args:
        subset: 'train', 'validation', 'test', or None for all
        streaming: If True, use streaming mode (doesn't download to disk)
    """
    from datasets import load_dataset
    
    logger.info(f"Starting ImageNet-1k download to: {IMAGENET_DIR}")
    logger.info(f"HF cache directory: {HF_CACHE_DIR}")
    
    os.makedirs(IMAGENET_DIR, exist_ok=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # Dataset configuration
    dataset_name = "ILSVRC/imagenet-1k"
    
    if streaming:
        logger.info("Using streaming mode (data will not be fully downloaded)")
        dataset = load_dataset(
            dataset_name,
            split=subset,
            streaming=True,
            token=HF_TOKEN
        )
        # Just verify streaming works
        logger.info("Streaming dataset loaded successfully")
        sample = next(iter(dataset))
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Sample label: {sample['label']}")
        return dataset
    
    # Full download
    splits_to_download = [subset] if subset else ['train', 'validation', 'test']
    
    for split in splits_to_download:
        logger.info(f"Downloading {split} split...")
        split_start = time.time()
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                token=HF_TOKEN,
                cache_dir=HF_CACHE_DIR,
                num_proc=4  # Parallel processing
            )
            
            split_duration = time.time() - split_start
            logger.info(f"Downloaded {split} split in {split_duration:.2f} seconds")
            logger.info(f"  - Number of samples: {len(dataset)}")
            logger.info(f"  - Features: {dataset.features}")
            
            # Save to disk in Arrow format for fast loading
            split_path = os.path.join(IMAGENET_DIR, split)
            logger.info(f"Saving {split} to {split_path}...")
            dataset.save_to_disk(split_path)
            logger.info(f"Saved {split} split to disk")
            
        except Exception as e:
            logger.error(f"Failed to download {split} split: {e}")
            raise
    
    total_duration = time.time() - start_time
    logger.info(f"Total download time: {total_duration/60:.2f} minutes")
    
    # Report disk usage
    logger.info("\nDisk usage:")
    for split in splits_to_download:
        split_path = os.path.join(IMAGENET_DIR, split)
        if os.path.exists(split_path):
            size = sum(f.stat().st_size for f in Path(split_path).rglob('*') if f.is_file())
            logger.info(f"  {split}: {size/1e9:.2f} GB")
    
    return IMAGENET_DIR


def download_imagenet_parquet(subset=None):
    """
    Alternative: Download ImageNet as parquet files directly.
    This can be faster for large datasets.
    """
    from huggingface_hub import snapshot_download
    
    logger.info("Downloading ImageNet-1k parquet files...")
    
    os.makedirs(IMAGENET_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # Download the parquet files
    snapshot_download(
        repo_id="ILSVRC/imagenet-1k",
        repo_type="dataset",
        local_dir=IMAGENET_DIR,
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
        # Only download parquet files for specified split
        allow_patterns=["*.parquet", "*.json", "*.md"] if subset is None 
                       else [f"*{subset}*.parquet", "*.json", "*.md"]
    )
    
    duration = time.time() - start_time
    logger.info(f"Download completed in {duration/60:.2f} minutes")
    
    # Report disk usage
    size = sum(f.stat().st_size for f in Path(IMAGENET_DIR).rglob('*') if f.is_file())
    logger.info(f"Total size: {size/1e9:.2f} GB")
    
    return IMAGENET_DIR


def verify_download():
    """Verify the downloaded dataset."""
    from datasets import load_from_disk
    
    logger.info("\nVerifying downloaded dataset...")
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(IMAGENET_DIR, split)
        if os.path.exists(split_path):
            try:
                dataset = load_from_disk(split_path)
                logger.info(f"{split}: {len(dataset)} samples loaded successfully")
                
                # Check a sample
                sample = dataset[0]
                logger.info(f"  Sample image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
                logger.info(f"  Sample label: {sample['label']}")
            except Exception as e:
                logger.error(f"Failed to verify {split}: {e}")
        else:
            logger.warning(f"{split} split not found at {split_path}")


def main():
    parser = argparse.ArgumentParser(description='Download ImageNet-1k from Hugging Face')
    parser.add_argument('--streaming', action='store_true', 
                        help='Use streaming mode (for testing, does not download full dataset)')
    parser.add_argument('--subset', type=str, choices=['train', 'validation', 'test'],
                        help='Download only specific subset')
    parser.add_argument('--parquet', action='store_true',
                        help='Download as parquet files (alternative method)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing download')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ImageNet-1k Download Script")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"ImageNet directory: {IMAGENET_DIR}")
    logger.info(f"HF cache: {HF_CACHE_DIR}")
    logger.info("=" * 60)
    
    # Setup authentication
    setup_authentication()
    
    if args.verify_only:
        verify_download()
        return
    
    # Download
    if args.parquet:
        download_imagenet_parquet(subset=args.subset)
    else:
        download_imagenet_hf(subset=args.subset, streaming=args.streaming)
    
    # Verify
    if not args.streaming:
        verify_download()
    
    logger.info("\nImageNet-1k download complete!")
    logger.info(f"Data stored in: {IMAGENET_DIR}")


if __name__ == "__main__":
    main()
