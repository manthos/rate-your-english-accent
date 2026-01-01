#!/usr/bin/env python3
"""
train.py - Standalone Training Script for Rate Your English Accent CNN

This script trains a ResNet18-based CNN to classify speech accents as English or Non-English.
It includes three experimental configurations with different regularization strategies.

Usage:
    # Train all experiments
    python train.py --data-path /path/to/dataset --experiment all
    
    # Train specific experiment
    python train.py --data-path /path/to/dataset --experiment exp3_optimized
    
    # Resume from checkpoint
    python train.py --data-path /path/to/dataset --experiment exp3_optimized --resume exp3_optimized_checkpoint_epoch12.pth
    
    # Quick test (5 epochs)
    python train.py --data-path /path/to/dataset --experiment exp1_baseline --quick-test

Requirements:
    pip install torch torchvision torchaudio numpy pandas scikit-learn tqdm matplotlib seaborn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR

import numpy as np
import pandas as pd
import os
import random
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
SEED = 42
ENGLISH_IDENTIFIERS = [
    'english', 'american', 'british', 'australian', 'canadian',
    'united states', 'united kingdom', 'australia', 'canada',
    'scotland', 'scottish', 'ireland', 'irish', 'welsh',
    'newzealand', 'new zealand', 'southafrican', 'south african'
]

# Preprocessing parameters (ImageNet normalization)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def is_english_accent(accent_value):
    """Binary classification: 1=English, 0=Non-English"""
    if pd.isna(accent_value):
        return 0
    accent_str = str(accent_value).lower().strip()
    return 1 if any(identifier in accent_str for identifier in ENGLISH_IDENTIFIERS) else 0

def resolve_npy_path(parquet_path, dataset_path, npy_files_cache):
    """Resolve parquet file path to actual .npy file location"""
    possible_paths = [
        parquet_path,
        os.path.join(dataset_path, parquet_path),
        os.path.join(dataset_path, os.path.basename(parquet_path)),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    basename = os.path.basename(parquet_path)
    for actual_file in npy_files_cache:
        if basename in actual_file:
            return actual_file
    return None

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
class SpecAugment:
    """SpecAugment: Frequency and time masking for spectrograms"""
    def __init__(self, freq_mask_param=10, time_mask_param=10, num_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
    
    def __call__(self, spec):
        spec = spec.clone()
        _, h, w = spec.shape
        
        # Frequency masking
        for _ in range(self.num_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, h - f))
            spec[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, w - t))
            spec[:, :, t0:t0+t] = 0
        
        return spec

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ============================================================================
# DATASET
# ============================================================================
class AccentDataset(Dataset):
    """Dataset for accent classification from pre-computed mel-spectrograms"""
    def __init__(self, dataframe, file_col, dataset_path, npy_cache, 
                 transform=None, augment=None, mode='train'):
        self.df = dataframe.reset_index(drop=True)
        self.file_col = file_col
        self.dataset_path = dataset_path
        self.npy_cache = npy_cache
        self.transform = transform
        self.augment = augment
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.df.loc[idx, self.file_col]
        resolved_path = resolve_npy_path(file_path, self.dataset_path, self.npy_cache)
        
        if resolved_path is None:
            return torch.zeros(3, 224, 224), 0
        
        try:
            mel_spec = np.load(resolved_path)
        except:
            return torch.zeros(3, 224, 224), 0
        
        # Convert to tensor
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Handle shape: (1, n_mels, time) → (n_mels, time)
        if len(mel_spec.shape) == 3 and mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0)
        
        # Add channel dimension if 2D
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # Replicate to 3 channels (RGB)
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.repeat(3, 1, 1)
        
        # Resize to 224x224
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(224, 224), 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Apply augmentation (training only)
        if self.augment and self.mode == 'train':
            mel_spec = self.augment(mel_spec)
        
        # Apply normalization
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        label = int(self.df.loc[idx, 'is_english'])
        return mel_spec, label

# ============================================================================
# MODEL
# ============================================================================
class AccentCNN(nn.Module):
    """ResNet18-based CNN for accent classification"""
    def __init__(self, dropout=0.5, freeze_layers=True):
        super(AccentCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers (transfer learning)
        if freeze_layers:
            for param in self.resnet.conv1.parameters():
                param.requires_grad = False
            for param in self.resnet.bn1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer2.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, config, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if config['mixup_alpha'] > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, config['mixup_alpha'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Step OneCycleLR per batch
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend((labels_a if config['mixup_alpha'] > 0 else labels).cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': epoch_loss, 'accuracy': epoch_acc, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
        'predictions': all_preds, 'labels': all_labels, 'probabilities': all_probs
    }

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================
CONFIGS = {
    'exp1_baseline': {
        'name': 'Experiment 1: Baseline (Conservative)',
        'batch_size': 32,
        'augment_params': None,
        'mixup_alpha': 0.0,
        'label_smoothing': 0.0,
        'dropout': 0.3,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'epochs': 30,
        'patience': 5,
        'checkpoint_every': 6
    },
    'exp2_regularized': {
        'name': 'Experiment 2: Regularized (Balanced)',
        'batch_size': 64,
        'augment_params': {'freq_mask_param': 22, 'time_mask_param': 22, 'num_masks': 2},
        'mixup_alpha': 0.2,
        'label_smoothing': 0.0,
        'dropout': 0.5,
        'lr': 5e-5,
        'weight_decay': 1e-3,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'epochs': 40,
        'patience': 7,
        'checkpoint_every': 6
    },
    'exp3_optimized': {
        'name': 'Experiment 3: Optimized (Aggressive)',
        'batch_size': 32,
        'augment_params': {'freq_mask_param': 34, 'time_mask_param': 34, 'num_masks': 2},
        'mixup_alpha': 0.3,
        'label_smoothing': 0.1,
        'dropout': 0.7,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 50,
        'patience': 10,
        'checkpoint_every': 6
    }
}

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_model(config, exp_name, train_df, val_df, test_df, dataset_path, 
                npy_files_cache, file_col, device, resume_from=None):
    """Main training loop for one experiment"""
    
    print(f"\n{'='*80}\n🎯 {config['name']}\n{'='*80}")
    
    # Create augmentation
    augment = None
    if config['augment_params']:
        augment = SpecAugment(**config['augment_params'])
    
    # Create datasets
    normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    
    train_dataset = AccentDataset(
        train_df, file_col, dataset_path, npy_files_cache, 
        transform=normalize, augment=augment, mode='train'
    )
    val_dataset = AccentDataset(
        val_df, file_col, dataset_path, npy_files_cache,
        transform=normalize, augment=None, mode='val'
    )
    test_dataset = AccentDataset(
        test_df, file_col, dataset_path, npy_files_cache,
        transform=normalize, augment=None, mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Create model
    model = AccentCNN(dropout=config['dropout'], freeze_layers=True).to(device)
    
    # Loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing']) \
                if config['label_smoothing'] > 0 else nn.CrossEntropyLoss()
    
    # Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Scheduler
    if config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    else:  # OneCycleLR
        scheduler = OneCycleLR(
            optimizer, max_lr=config['lr'], 
            epochs=config['epochs'], steps_per_epoch=len(train_loader)
        )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_roc_auc': []
    }
    best_val_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0)
        history = checkpoint.get('history', history)
        print(f"   Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    
    best_epoch = start_epoch
    patience_counter = 0
    
    print(f"\n🏋️  Training for {config['epochs']} epochs...")
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{config['epochs']}\n{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config,
            scheduler if isinstance(scheduler, OneCycleLR) else None
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        print(f"\n📊 Metrics:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Scheduler step
        if config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_metrics['loss'])
        elif config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Learning rate: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'history': history
            }, f'{exp_name}_best_model.pth')
            
            print(f"   ✅ New best model! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{config['patience']}")
        
        # Checkpoint
        if (epoch + 1) % config['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'history': history
            }, f'{exp_name}_checkpoint_epoch{epoch+1}.pth')
            print(f"   💾 Checkpoint saved")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for testing
    print(f"\n🔍 Loading best model (epoch {best_epoch})...")
    checkpoint = torch.load(f'{exp_name}_best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_metrics = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    print(f"\n{'='*60}\n📊 FINAL TEST RESULTS\n{'='*60}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    return {
        'config': config,
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics
    }

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train CNN for accent classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'exp1_baseline', 'exp2_regularized', 'exp3_optimized'],
                        help='Which experiment to run')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test: train for 5 epochs only')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("🚀 ACCENT CLASSIFICATION - CNN TRAINING")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Experiment: {args.experiment}")
    
    # Set seed
    set_seed(args.seed)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    parquet_files = [f for f in os.listdir(args.data_path) if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {args.data_path}")
    
    df = pd.read_parquet(os.path.join(args.data_path, parquet_files[0]))
    print(f"   Loaded {len(df)} samples")
    
    # Auto-detect columns
    file_col = next((col for col in ['mel_spectrogram_path', 'file_path', 'filename', 'path'] 
                     if col in df.columns), None)
    accent_col = next((col for col in ['accent', 'accent_id', 'label', 'native_language'] 
                       if col in df.columns), None)
    
    if not file_col or not accent_col:
        raise ValueError(f"Could not detect file or accent column. Available: {df.columns.tolist()}")
    
    print(f"   File column: {file_col}")
    print(f"   Accent column: {accent_col}")
    
    # Create binary labels
    df['is_english'] = df[accent_col].apply(is_english_accent)
    print(f"   English: {df['is_english'].sum()}, Non-English: {(~df['is_english'].astype(bool)).sum()}")
    
    # Build .npy cache
    print(f"\n🔍 Building .npy file cache...")
    npy_files_cache = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.endswith('.npy'):
                npy_files_cache.append(os.path.join(root, file))
    print(f"   Found {len(npy_files_cache)} .npy files")
    
    # Split dataset
    print(f"\n✂️  Splitting dataset...")
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['is_english']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=args.seed, stratify=train_val_df['is_english']
    )
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Modify epochs if quick test
    if args.quick_test:
        print(f"\n⚡ QUICK TEST MODE: Reducing to 5 epochs")
        for config in CONFIGS.values():
            config['epochs'] = 5
            config['patience'] = 2
    
    # Run experiments
    results = {}
    experiments_to_run = [args.experiment] if args.experiment != 'all' else list(CONFIGS.keys())
    
    for exp_name in experiments_to_run:
        if exp_name not in CONFIGS:
            print(f"\n⚠️  Unknown experiment: {exp_name}")
            continue
        
        config = CONFIGS[exp_name]
        try:
            result = train_model(
                config, exp_name, train_df, val_df, test_df,
                args.data_path, npy_files_cache, file_col, device,
                resume_from=args.resume
            )
            results[exp_name] = result
        except Exception as e:
            print(f"\n❌ Error in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    if results:
        print(f"\n{'='*80}\n📊 TRAINING COMPLETE\n{'='*80}")
        
        summary = {
            exp: {
                'test_accuracy': float(res['test_metrics']['accuracy']),
                'test_f1': float(res['test_metrics']['f1']),
                'test_roc_auc': float(res['test_metrics']['roc_auc']),
                'best_val_acc': float(res['best_val_acc']),
                'best_epoch': int(res['best_epoch'])
            }
            for exp, res in results.items()
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Saved training_results.json")
        
        # Print comparison
        comparison_data = []
        for exp_name, res in results.items():
            comparison_data.append({
                'Experiment': CONFIGS[exp_name]['name'],
                'Test Acc': f"{res['test_metrics']['accuracy']:.4f}",
                'Test F1': f"{res['test_metrics']['f1']:.4f}",
                'Test AUC': f"{res['test_metrics']['roc_auc']:.4f}",
                'Best Val Acc': f"{res['best_val_acc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n{comparison_df.to_string(index=False)}")
        
        best_exp = max(results.items(), key=lambda x: x[1]['test_metrics']['accuracy'])
        print(f"\n🏆 BEST: {CONFIGS[best_exp[0]]['name']}")
        print(f"   Test Accuracy: {best_exp[1]['test_metrics']['accuracy']:.4f}")
    
    print(f"\n✅ Training complete!")

if __name__ == '__main__':
    main()
