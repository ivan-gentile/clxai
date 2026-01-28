"""Supervised Contrastive Learning for ResNet on ImageNet."""
import os, sys, argparse, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.imagenet_resnet import get_imagenet_resnet
from src.models.classifiers import KNNClassifier, LinearClassifier, train_linear_classifier
from src.training.losses import SupConLossV2, TripletLoss
from src.utils.imagenet_data import get_imagenet_loaders

try:
    import wandb; WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device, epoch, loss_type='supcon', grad_clip=None):
    model.train(); total_loss = 0.0
    # Handle DataParallel wrapper for get_embedding
    model_ref = model.module if hasattr(model, 'module') else model
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        if isinstance(images, list): images = torch.cat([images[0], images[1]], dim=0).to(device)
        else: images = images.to(device)
        labels = labels.to(device); optimizer.zero_grad()
        embeddings = model_ref.get_embedding(images, normalize=False) if loss_type == 'triplet' else model(images)
        loss = criterion(embeddings, labels); loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step(); total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return {'train_loss': total_loss / len(loader)}

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=None):
    model.eval(); embeddings, labels, total = [], [], 0
    # Handle DataParallel wrapper
    model_ref = model.module if hasattr(model, 'module') else model
    for images, targets in tqdm(loader, desc="Extracting"):
        if isinstance(images, list): images = images[0]
        features = model_ref.get_embedding(images.to(device), normalize=True)
        embeddings.append(features.cpu()); labels.append(targets)
        total += images.size(0)
        if max_samples and total >= max_samples: break
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

def evaluate_knn(model, train_loader, test_loader, device, k=200, max_train=100000):
    train_emb, train_labels = extract_embeddings(model, train_loader, device, max_train)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    knn = KNNClassifier(k=k, metric='cosine'); knn.fit(train_emb, train_labels)
    return knn.score(test_emb, test_labels) * 100

def train_scl_imagenet(config):
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config.get('output_dir', 'results/models/supcon_imagenet'))
    output_dir.mkdir(parents=True, exist_ok=True)
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(project=config.get('wandb_project', 'clxai'), name=config.get('run_name'), config=config)
    loss_type, aug_type = config.get('loss', 'supcon'), config.get('augmentation_type', 'none')
    data_dir = config.get('data_dir', '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k')
    train_loader, val_loader = get_imagenet_loaders(data_dir=data_dir, batch_size=config.get('batch_size', 256),
        num_workers=config.get('num_workers', 16), contrastive=(loss_type=='supcon'), augment=True, augmentation_type=aug_type)
    train_loader_eval, _ = get_imagenet_loaders(data_dir=data_dir, batch_size=config.get('batch_size', 256),
        num_workers=config.get('num_workers', 16), contrastive=False, augment=False)
    model = get_imagenet_resnet(config.get('architecture', 'resnet152_imagenet'), 1000, True, config.get('embedding_dim', 128)).to(device)
    feature_dim = model.feature_dim if hasattr(model, 'feature_dim') else 2048
    # Multi-GPU support with DataParallel
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {feature_dim}D features")
    criterion = SupConLossV2(temperature=config.get('temperature', 0.1)) if loss_type == 'supcon' else TripletLoss(margin=0.3, mining='semi-hard')
    optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 0.5), momentum=0.9, weight_decay=config.get('weight_decay', 1e-4))
    epochs, warmup = config.get('epochs', 350), config.get('warmup_epochs', 10)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.1+0.9*(e/warmup) if e<warmup else 0.5*(1+np.cos(np.pi*(e-warmup)/(epochs-warmup))))
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    resume_path = config.get('resume')
    if resume_path and Path(resume_path).exists():
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
        print(f"Resumed from epoch {start_epoch-1}, best_acc={best_acc:.2f}%")
    
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, loss_type, config.get('grad_clip'))
        scheduler.step()
        if epoch % config.get('eval_freq', 10) == 0 or epoch == epochs:
            knn_acc = evaluate_knn(model, train_loader_eval, val_loader, device, k=config.get('knn_k', 200))
            metrics['knn_acc'] = knn_acc
            print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, kNN={knn_acc:.2f}%, time={time.time()-t0:.0f}s")
            if knn_acc > best_acc:
                best_acc = knn_acc
                # Save model without DataParallel wrapper
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict(), 'knn_acc': best_acc}, output_dir / 'best_model.pt')
        else:
            print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, time={time.time()-t0:.0f}s")
        if WANDB_AVAILABLE and config.get('use_wandb', True): wandb.log(metrics, step=epoch)
        if epoch % config.get('save_freq', 50) == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'knn_acc': best_acc
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint: checkpoint_epoch_{epoch}.pt")
            # Keep only last 3 checkpoints to save disk space (~2 GB instead of 24 GB)
            old_ckpts = sorted(output_dir.glob('checkpoint_epoch_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))[:-3]
            for old_ckpt in old_ckpts:
                old_ckpt.unlink()
                print(f"Removed old checkpoint: {old_ckpt.name}")
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({'epoch': epochs, 'model_state_dict': model_to_save.state_dict()}, output_dir / 'final_model.pt')
    print(f"\nTraining linear classifier...")
    train_emb, train_labels = extract_embeddings(model, train_loader_eval, device, 500000)
    test_emb, test_labels = extract_embeddings(model, val_loader, device)
    linear_clf = LinearClassifier(input_dim=feature_dim, num_classes=1000)
    history = train_linear_classifier(linear_clf, torch.tensor(train_emb), torch.tensor(train_labels),
        torch.tensor(test_emb), torch.tensor(test_labels), epochs=100, lr=0.1, device=str(device))
    linear_acc = history['val_acc'][-1] * 100
    print(f"Linear probe: {linear_acc:.2f}%")
    torch.save(linear_clf.state_dict(), output_dir / 'linear_classifier.pt')
    print(f"\nDone in {(time.time()-start)/3600:.2f}h. Best kNN: {best_acc:.2f}%, Linear: {linear_acc:.2f}%")
    if WANDB_AVAILABLE and config.get('use_wandb', True): wandb.finish()
    return model, best_acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str); p.add_argument('--architecture', default='resnet152_imagenet')
    p.add_argument('--epochs', type=int, default=350); p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.5); p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.1); p.add_argument('--loss', default='supcon')
    p.add_argument('--data_dir', default='/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k')
    p.add_argument('--output_dir', default='results/models/supcon_imagenet')
    p.add_argument('--no_wandb', action='store_true'); p.add_argument('--run_name', default='supcon_imagenet_r152')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--augmentation_type', default='none')
    p.add_argument('--num_workers', type=int, default=16); p.add_argument('--eval_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=50); p.add_argument('--knn_k', type=int, default=200)
    p.add_argument('--grad_clip', type=float, default=None)
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = p.parse_args()
    if args.config and Path(args.config).exists():
        with open(args.config) as f: config = yaml.safe_load(f)
        for sec in ['model','training','data','evaluation','logging','output']:
            if sec in config:
                for k,v in config[sec].items(): config[k] = v
        if 'augmentation' in config: config['augmentation_type'] = config['augmentation'].get('type', 'none')
    else:
        config = vars(args).copy()
        config['use_wandb'] = not args.no_wandb; config['wandb_project'] = 'clxai'
        config['warmup_epochs'] = 10; config['embedding_dim'] = 128
    # Command line resume overrides config
    if args.resume:
        config['resume'] = args.resume
    train_scl_imagenet(config)

if __name__ == "__main__": main()
