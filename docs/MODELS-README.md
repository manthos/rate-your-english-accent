# Pre-trained Models

This directory contains pre-trained model weights for the English Accent Classifier.

## Download Pre-trained Models

Due to file size limitations on GitHub, the pre-trained models are hosted externally.

### Option 1: Download from Kaggle (Recommended)

Download all pre-trained models from the Kaggle notebook output:

```bash
# Install Kaggle CLI
pip install kaggle

# Download the notebook output (contains all models)
kaggle kernels output websaasai/speechaccentkaggle -p ./kaggle-models

# Copy the best model
cp ./kaggle-models/exp3_optimized_best_model.pth ./models/
cp ./kaggle-models/accent_classifier_traced.pt ./models/
```

### Option 2: Train Your Own Model

You can train the model from scratch:

```bash
# Train with default configuration (exp3_optimized)
python train.py --data_path ./data/final_segmented_metadata_with_mels.parquet

# The trained model will be saved to ./models/
```

Training takes approximately 2-3 hours on a GPU (NVIDIA T4 or better).

## Available Model Files

After downloading, you should have:

### 1. Production Model (Required for Inference)
- **exp3_optimized_best_model.pth** (44.2 MB)
  - Best performing model from experiment 3
  - Test Accuracy: 89.56%
  - Validation Accuracy: 90.36%
  - Format: PyTorch state dictionary

### 2. TorchScript Traced Model (Alternative)
- **accent_classifier_traced.pt** (44.5 MB)
  - TorchScript traced version for deployment
  - Same accuracy as above
  - Can be loaded without model architecture code

### 3. Experimental Models (Optional)
- **exp1_baseline_best_model.pth** - Baseline experiment (74.8% test)
- **exp2_regularized_best_model.pth** - With regularization (75.9% test)
- Various checkpoint files for each experiment

## Model Architecture

- **Base**: ResNet18 (pre-trained on ImageNet)
- **Input**: Mel-spectrogram images (3, 224, 224)
- **Output**: Binary classification (English / Non-English)
- **Framework**: PyTorch 2.1.0

## Model Configuration

The model uses these hyperparameters (from exp3_optimized):

```json
{
  "dropout_rate": 0.7,
  "learning_rate": 0.0001,
  "batch_size": 32,
  "epochs": 50,
  "early_stopping_patience": 5,
  "lr_scheduler": "ReduceLROnPlateau",
  "augmentation": true
}
```

## Usage

### Load Model for Inference

```python
from model_class import AccentCNN
import torch

# Load the model
model = AccentCNN(dropout_rate=0.7)
model.load_state_dict(torch.load('./models/exp3_optimized_best_model.pth'))
model.eval()

# Or use predict.py
python predict.py --audio_path sample.wav --model_path ./models/exp3_optimized_best_model.pth
```

### Use TorchScript Model

```python
import torch

# Load traced model (no need for model architecture)
model = torch.jit.load('./models/accent_classifier_traced.pt')
model.eval()
```

## Model Performance

| Experiment | Dropout | LR Scheduler | Augmentation | Val Acc | Test Acc |
|------------|---------|--------------|--------------|---------|----------|
| exp1_baseline | 0.3 | None | No | 75.2% | 74.8% |
| exp2_regularized | 0.5 | StepLR | Yes | 76.4% | 75.9% |
| **exp3_optimized** | **0.7** | **ReduceLR** | **Yes** | **90.36%** | **89.56%** |

### Detailed Metrics (exp3_optimized)

- **Accuracy**: 89.56%
- **ROC-AUC**: 0.9382
- **Precision (Non-English)**: 91.84%
- **Precision (English)**: 80.53%
- **Recall (Non-English)**: 93.51%
- **Recall (English)**: 75.75%
- **F1-Score (Non-English)**: 0.9267
- **F1-Score (English)**: 0.7807

## File Structure

```
models/
├── README.md (this file)
├── exp3_optimized_best_model.pth (download required)
├── accent_classifier_traced.pt (download required)
└── .gitkeep
```

## Notes

- Models are not committed to git due to size (>50MB each)
- Use Git LFS if you want to commit models to your fork
- The `.gitignore` excludes `*.pth` and `*.pt` files by default
