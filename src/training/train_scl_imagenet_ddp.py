"""Supervised Contrastive Learning for ResNet on ImageNet - Multi-Node DDP Version."""
import os, sys, argparse, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.imagenet_resnet import get_imagenet_resnet
from src.models.classifiers import KNNClassifier, LinearClassifier, train_linear_classifier
from src.training.losses import SupConLossV2, TripletLoss
from src.utils.imagenet_data import get_imagenet_loaders, get_imagenet_datasets

try:
    import wandb; WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def setup_distributed():
    """Initialize distributed training from SLURM environment."""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        # Get master address from first node
        node_list = os.environ.get('SLURM_NODELIST', '')
        if node_list:
            import subprocess
            result = subprocess.run(['scontrol', 'show', 'hostnames', node_list], 
                                  capture_output=True, text=True)
            master_addr = result.stdout.strip().split('\n')[0]
        else:
            master_addr = 'localhost'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        # If SLURM/srun sets CUDA_VISIBLE_DEVICES per task, use device 0
        # Otherwise use local_rank
        n_visible_gpus = torch.cuda.device_count()
        if n_visible_gpus == 1:
            # SLURM gave us exactly 1 GPU, use it
            device_id = 0
        else:
            # Multiple GPUs visible, select by local_rank
            device_id = local_rank
    else:
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device_id = local_rank
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://',
                               rank=rank, world_size=world_size)
    
    torch.cuda.set_device(device_id)
    
    # Return device_id as local_rank for model placement
    return rank, device_id, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def train_epoch(model, loader, criterion, optimizer, device, epoch, loss_type='supcon', grad_clip=None, rank=0):
    model.train()
    total_loss = 0.0
    # Handle DDP wrapper for get_embedding
    model_ref = model.module if hasattr(model, 'module') else model
    
    if is_main_process(rank):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
    else:
        pbar = loader
    
    for images, labels in pbar:
        if isinstance(images, list):
            images = torch.cat([images[0], images[1]], dim=0).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        if loss_type == 'triplet':
            embeddings = model_ref.get_embedding(images, normalize=False)
        else:
            embeddings = model(images)
        
        loss = criterion(embeddings, labels)
        loss.backward()
        
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        
        if is_main_process(rank) and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'train_loss': total_loss / len(loader)}

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=None):
    model.eval()
    embeddings, labels, total = [], [], 0
    model_ref = model.module if hasattr(model, 'module') else model
    
    for images, targets in loader:
        if isinstance(images, list):
            images = images[0]
        features = model_ref.get_embedding(images.to(device), normalize=True)
        embeddings.append(features.cpu())
        labels.append(targets)
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

def evaluate_knn(model, train_loader, test_loader, device, k=200, max_train=100000, rank=0):
    """Evaluate on main process only."""
    if not is_main_process(rank):
        return 0.0
    
    train_emb, train_labels = extract_embeddings(model, train_loader, device, max_train)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    knn = KNNClassifier(k=k, metric='cosine')
    knn.fit(train_emb, train_labels)
    return knn.score(test_emb, test_labels) * 100

def get_distributed_loaders(config, world_size, rank, loss_type='supcon'):
    """Create data loaders with DistributedSampler."""
    from src.utils.imagenet_data import get_imagenet_datasets
    
    data_dir = config.get('data_dir', '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k')
    aug_type = config.get('augmentation_type', 'none')
    batch_size = config.get('batch_size', 256)
    num_workers = config.get('num_workers', 8)
    
    # Get datasets
    train_dataset, val_dataset = get_imagenet_datasets(
        data_dir=data_dir,
        contrastive=(loss_type == 'supcon'),
        augment=True,
        augmentation_type=aug_type
    )
    train_dataset_eval, _ = get_imagenet_datasets(
        data_dir=data_dir,
        contrastive=False,
        augment=False
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_sampler_eval = DistributedSampler(train_dataset_eval, num_replicas=world_size, rank=rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    train_loader_eval = torch.utils.data.DataLoader(
        train_dataset_eval, batch_size=batch_size, sampler=train_sampler_eval,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, train_loader_eval, val_loader, train_sampler

def train_scl_imagenet_ddp(config):
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Debug: Print setup info from each rank
    import socket
    hostname = socket.gethostname()
    n_gpus = torch.cuda.device_count()
    print(f"[Rank {rank}] Host: {hostname}, LocalRank: {local_rank}, GPUs visible: {n_gpus}, Using device: {device}")
    
    set_seed(config.get('seed', 42), rank)
    
    output_dir = Path(config.get('output_dir', 'results/models/supcon_imagenet'))
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"=== Multi-Node DDP Training ===")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Batch per GPU: {config.get('batch_size', 128)}")
        print(f"Effective batch size: {config.get('batch_size', 128) * world_size}")
    
    # Synchronize before continuing
    if world_size > 1:
        dist.barrier()
    
    # Initialize wandb only on main process
    if is_main_process(rank) and WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(project=config.get('wandb_project', 'clxai'), 
                  name=config.get('run_name'), config=config)
    
    loss_type = config.get('loss', 'supcon')
    
    # Get distributed data loaders
    train_loader, train_loader_eval, val_loader, train_sampler = get_distributed_loaders(
        config, world_size, rank, loss_type
    )
    
    # Create model
    model = get_imagenet_resnet(
        config.get('architecture', 'resnet152_imagenet'), 
        1000, True, config.get('embedding_dim', 128)
    ).to(device)
    
    feature_dim = model.feature_dim if hasattr(model, 'feature_dim') else 2048
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {total_params/1e6:.1f}M params, {feature_dim}D features")
    
    # Loss function
    if loss_type == 'supcon':
        criterion = SupConLossV2(temperature=config.get('temperature', 0.1))
    else:
        criterion = TripletLoss(margin=0.3, mining='semi-hard')
    
    # Optimizer - scale LR based on effective batch size
    # Reference: SupCon paper uses LR=0.5 for batch=1024
    # Linear scaling rule: LR = base_lr * (effective_batch / reference_batch)
    base_lr = config.get('lr', 0.5)
    batch_size = config.get('batch_size', 128)
    effective_batch = batch_size * world_size
    reference_batch = 1024  # SupCon paper reference
    lr_scale = effective_batch / reference_batch
    scaled_lr = base_lr * lr_scale
    
    if is_main_process(rank):
        print(f"LR scaling: base={base_lr}, effective_batch={effective_batch}, scale={lr_scale:.2f}x, final_lr={scaled_lr:.3f}")
    
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, 
                         weight_decay=config.get('weight_decay', 1e-4))
    
    epochs = config.get('epochs', 350)
    warmup = config.get('warmup_epochs', 10)
    
    # Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda e: 0.1 + 0.9 * (e / warmup) if e < warmup else 
                  0.5 * (1 + np.cos(np.pi * (e - warmup) / (epochs - warmup)))
    )
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    resume_path = config.get('resume')
    if resume_path and Path(resume_path).exists():
        if is_main_process(rank):
            print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model_ref = model.module if hasattr(model, 'module') else model
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('knn_acc', 0.0)
        if is_main_process(rank):
            print(f"Resumed from epoch {start_epoch-1}, best_acc={best_acc:.2f}%")
    
    start = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        t0 = time.time()
        metrics = train_epoch(model, train_loader, criterion, optimizer, device, 
                            epoch, loss_type, config.get('grad_clip'), rank)
        scheduler.step()
        
        # Evaluation (only on main process, or gather embeddings)
        if epoch % config.get('eval_freq', 10) == 0 or epoch == epochs:
            # For kNN evaluation, we need non-distributed loaders on main process
            if is_main_process(rank):
                # Re-create eval loaders without distributed sampler
                data_dir = config.get('data_dir')
                train_loader_eval_single, val_loader_single = get_imagenet_loaders(
                    data_dir=data_dir, batch_size=config.get('batch_size', 256),
                    num_workers=config.get('num_workers', 8), contrastive=False, augment=False
                )
                knn_acc = evaluate_knn(model, train_loader_eval_single, val_loader_single, 
                                      device, k=config.get('knn_k', 200), rank=rank)
                metrics['knn_acc'] = knn_acc
                print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, kNN={knn_acc:.2f}%, time={time.time()-t0:.0f}s")
                
                if knn_acc > best_acc:
                    best_acc = knn_acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'epoch': epoch, 
                        'model_state_dict': model_to_save.state_dict(), 
                        'knn_acc': best_acc
                    }, output_dir / 'best_model.pt')
        else:
            if is_main_process(rank):
                print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, time={time.time()-t0:.0f}s")
        
        # Logging
        if is_main_process(rank) and WANDB_AVAILABLE and config.get('use_wandb', True):
            wandb.log(metrics, step=epoch)
        
        # Save checkpoint (only main process)
        if is_main_process(rank) and epoch % config.get('save_freq', 10) == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'knn_acc': best_acc
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint: checkpoint_epoch_{epoch}.pt")
            
            # Keep only last 3 checkpoints
            old_ckpts = sorted(output_dir.glob('checkpoint_epoch_*.pt'), 
                             key=lambda p: int(p.stem.split('_')[-1]))[:-3]
            for old_ckpt in old_ckpts:
                old_ckpt.unlink()
                print(f"Removed old checkpoint: {old_ckpt.name}")
        
        # Synchronize after checkpointing
        if world_size > 1:
            dist.barrier()
    
    # Final model and linear probe (main process only)
    if is_main_process(rank):
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epochs, 'model_state_dict': model_to_save.state_dict()}, 
                  output_dir / 'final_model.pt')
        
        print(f"\nTraining linear classifier...")
        data_dir = config.get('data_dir')
        train_loader_eval_single, val_loader_single = get_imagenet_loaders(
            data_dir=data_dir, batch_size=config.get('batch_size', 256),
            num_workers=config.get('num_workers', 8), contrastive=False, augment=False
        )
        train_emb, train_labels = extract_embeddings(model, train_loader_eval_single, device, 500000)
        test_emb, test_labels = extract_embeddings(model, val_loader_single, device)
        
        linear_clf = LinearClassifier(input_dim=feature_dim, num_classes=1000)
        history = train_linear_classifier(
            linear_clf, torch.tensor(train_emb), torch.tensor(train_labels),
            torch.tensor(test_emb), torch.tensor(test_labels), 
            epochs=100, lr=0.1, device=str(device)
        )
        linear_acc = history['val_acc'][-1] * 100
        print(f"Linear probe: {linear_acc:.2f}%")
        torch.save(linear_clf.state_dict(), output_dir / 'linear_classifier.pt')
        
        print(f"\nDone in {(time.time()-start)/3600:.2f}h. Best kNN: {best_acc:.2f}%, Linear: {linear_acc:.2f}%")
        
        if WANDB_AVAILABLE and config.get('use_wandb', True):
            wandb.finish()
    
    cleanup_distributed()
    return best_acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str)
    p.add_argument('--architecture', default='resnet152_imagenet')
    p.add_argument('--epochs', type=int, default=350)
    p.add_argument('--batch_size', type=int, default=128)  # Per GPU
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--loss', default='supcon')
    p.add_argument('--data_dir', default='/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k')
    p.add_argument('--output_dir', default='results/models/supcon_imagenet')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--run_name', default='supcon_imagenet_r152')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augmentation_type', default='none')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--eval_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--knn_k', type=int, default=200)
    p.add_argument('--grad_clip', type=float, default=None)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for sec in ['model', 'training', 'data', 'evaluation', 'logging', 'output']:
            if sec in config:
                for k, v in config[sec].items():
                    config[k] = v
        if 'augmentation' in config:
            config['augmentation_type'] = config['augmentation'].get('type', 'none')
    else:
        config = vars(args).copy()
        config['use_wandb'] = not args.no_wandb
        config['wandb_project'] = 'clxai'
        config['warmup_epochs'] = 10
        config['embedding_dim'] = 128
    
    if args.resume:
        config['resume'] = args.resume
    
    train_scl_imagenet_ddp(config)

if __name__ == "__main__":
    main()
