# train.py Usage Guide

## 🚀 Quick Start

```bash
# Train all experiments (recommended)
python train.py --data-path /kaggle/input/speech-accent --experiment all

# Train specific experiment
python train.py --data-path /kaggle/input/speech-accent --experiment exp3_optimized

# Quick test (5 epochs)
python train.py --data-path /kaggle/input/speech-accent --experiment exp1_baseline --quick-test
```

## 📋 Requirements

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn tqdm matplotlib seaborn
```

## 🎯 Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-path` | ✅ Yes | - | Path to dataset directory |
| `--experiment` | ❌ No | all | Which experiment: `all`, `exp1_baseline`, `exp2_regularized`, `exp3_optimized` |
| `--resume` | ❌ No | None | Resume from checkpoint file |
| `--quick-test` | ❌ No | False | Quick test mode (5 epochs) |
| `--device` | ❌ No | auto | Device: `auto`, `cpu`, `cuda` |
| `--seed` | ❌ No | 42 | Random seed |

## 📊 Experiment Configurations

### Experiment 1: Baseline (Conservative)
- **Goal:** Establish CNN baseline
- **Augmentation:** None
- **Dropout:** 0.3
- **Epochs:** 30
- **Expected:** 75-76% accuracy

### Experiment 2: Regularized (Balanced)
- **Goal:** Combat overfitting
- **Augmentation:** SpecAugment + Mixup
- **Dropout:** 0.5
- **Epochs:** 40
- **Expected:** 76-78% accuracy

### Experiment 3: Optimized (Aggressive)
- **Goal:** Maximize generalization
- **Augmentation:** Full suite + Label Smoothing
- **Dropout:** 0.7
- **Epochs:** 50
- **Expected:** 78-80% accuracy

## 📁 Output Files

The script generates:
- `{exp}_best_model.pth` - Best model checkpoint
- `{exp}_checkpoint_epoch{N}.pth` - Periodic checkpoints
- `training_results.json` - Summary of all experiments

## 🔄 Resume Training

If training is interrupted:

```bash
python train.py --data-path /kaggle/input/speech-accent \
    --experiment exp3_optimized \
    --resume exp3_optimized_checkpoint_epoch12.pth
```

## 🧪 Quick Test Mode

Test training pipeline quickly (5 epochs):

```bash
python train.py --data-path /kaggle/input/speech-accent --quick-test
```

## 📈 Monitoring Training

The script outputs:
- Train/Val loss and accuracy per epoch
- Learning rate schedule
- Best model updates
- Early stopping status

Example output:
```
============================================================
Epoch 12/50
============================================================
Training: 100%|██████████| 234/234 [02:15<00:00]
Validation: 100%|██████████| 59/59 [00:28<00:00]

📊 Metrics:
   Train Loss: 0.2156 | Train Acc: 0.9123
   Val Loss: 0.3842 | Val Acc: 0.7654
   Val F1: 0.7612 | Val ROC-AUC: 0.8234
   Learning rate: 5.23e-05
   ✅ New best model! Val Acc: 0.7654
```

## 🎯 Best Practices

1. **Always run all experiments** to compare performance
2. **Use quick-test first** to verify setup
3. **Monitor train-val gap** for overfitting
4. **Resume from checkpoints** if interrupted
5. **Compare test metrics** not just validation

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch_size or use CPU
```bash
python train.py --data-path /path --device cpu
```

### Issue: Dataset not found
**Solution:** Check data-path argument
```bash
ls /kaggle/input/speech-accent/*.parquet
```

### Issue: Training too slow
**Solution:** Reduce num_workers or use GPU
```python
# Edit train.py: num_workers=0
```

## ✅ After Training

1. Check `training_results.json` for comparison
2. Load best model for deployment
3. Run Cell 9 (Model Selection) to choose winner
4. Export with Cell 11 (TorchScript)

## 📦 Integration with Project

This script is standalone but integrates with:
- `predict.py` - Uses trained models
- `model_class.py` - Defines AccentCNN architecture
- `model_config.json` - Configuration metadata

Complete workflow:
```bash
# 1. Train
python train.py --data-path /data --experiment all

# 2. Select best
# (Review training_results.json)

# 3. Test inference
python predict.py --audio sample.wav --model exp3_optimized_best_model.pt
```
