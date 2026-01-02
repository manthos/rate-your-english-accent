# 🎙️ Rate Your English Accent

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange.svg)](https://gradio.app/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Original Dataset](https://img.shields.io/badge/Original-Speech%20Accent%20Archive-blue.svg)](https://huggingface.co/datasets/CodecSR/speech_accent_archive_synth)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/websaasai/speech-accent)

> **ML Zoomcamp 2025 Capstone Project** - Deep learning accent binary classifier using ResNet18 CNN on mel-spectrograms

**🎯 Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/manthos/rate-your-english-accent) | [GitHub Repo](https://github.com/manthos/rate-your-english-accent)

📊 For project evaluation criteria mapping, see [EVALUATION.md](./EVALUATION.md)

---

## 📋 Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Reproducibility](#reproducibility)
- [Model Deployment](#model-deployment)
- [Containerization](#containerization)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [License](#license)

---

## 🎯 Problem Description

### **The Challenge**

Accent classification is essential in modern speech processing applications. Distinguishing between native English speakers and non-native speakers helps improve:

- **Voice assistants** - Adapting speech recognition models for better accuracy
- **Language learning platforms** - Providing targeted pronunciation feedback
- **Call centers** - Intelligent call routing to appropriate support agents
- **Accessibility tools** - Enhanced transcription quality across diverse accents
- **Market research** - Understanding demographic patterns in speech data

### **Business Problem**

Organizations need automated, scalable solutions to:
- Reduce reliance on manual accent assessment (time-consuming and subjective)
- Improve customer experience through accent-aware services
- Enable real-time accent detection in production systems
- Support multilingual applications with accent-specific processing

### **Our Solution**

A **binary classifier** powered by deep learning that predicts:
- ✅ **English Accent** - Native or native-level English speakers (American)
- ❌ **Non-English Accent** - Speakers with other native languages

Binary classifier that detects whether a speaker has a native English accent or a non-English accent - not a measure of English proficiency.

### **Technical Approach**

1. **Input Processing**: Audio files (WAV/MP3/FLAC) converted to mel-spectrograms
2. **Feature Extraction**: 80 mel-frequency bins at 16kHz sample rate
3. **Model Architecture**: ResNet18 pre-trained on ImageNet, fine-tuned on speech spectrograms
4. **Classification**: Binary prediction with calibrated probabilities
5. **Deployment**: REST API via Gradio, containerized with Docker, hosted on HuggingFace

### **Key Features**

- **High Test Accuracy**: 89.6% test accuracy, 0.94 ROC-AUC
- **Calibrated Probabilities**: Adjusts for 3.4:1 class imbalance in training data
- **Real-time Inference**: Processes audio in seconds
- **Production-Ready**: Dockerized, cloud-deployed, API accessible
- **Transparent**: Shows both raw and calibrated confidence scores

---

## 📊 Dataset

### **Important Note for Evaluation**
This repository contains 3 notebooks not one per project suggestion.
- The first notebook is ./SpeechAccent_EDA.ipynb this is a Google Collab notebook that cannot be viewed on Github (known issue https://stackoverflow.com/questions/79661958/jupyter-notebook-rendering-error-state-key-missing-from-metadata-widgets-de ). This is where part of our EDA and the creation of our Kaggle dataset lies. I have shared it for the duration of the class so you can open in Google Collab directly if you like - use this link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link

- The second notebook is ./notebook.ipynb which is the same as ./SpeechAccent_EDA.ipynb but converted (as per stackoverflow link above) so it can display on Github (added metadata on notebook)

- The third notebook is ./speechaccentkaggle.ipynb. This a Kaggle notebook that we use to continue our EDA, do feature importance analysis, train the models and everything else for the project. (Our Google Collab GPU run out)

### **Source**

- **Original**: [Speech Accent Archive](https://accent.gmu.edu/) - George Mason University
- **Processed Dataset after our extensive EDA**: EDA Google Collab Notebook: (https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link) [Speech Accent Split and Mel-spectrograms](https://www.kaggle.com/datasets/websaasai/speech-accent) on Kaggle
- **Processing**: CodecSR-based silence removal, 7-second segmentation, mel-spectrogram pre-computation
- **License**: CC BY-NC-SA 4.0

### **Dataset Statistics**

| Metric | Value |
|--------|-------|
| **Total Samples** | 12,062 audio segments |
| **Test Set Size** | 2,413 samples |
| **Class Distribution** | 3.4:1 (Non-English:English) |
| **English Samples** | 550 test samples (22.8%) |
| **Non-English Samples** | 1,863 test samples (77.2%) |
| **Audio Format** | 16kHz mono, 7-second segments |
| **Feature Format** | 80-bin mel-spectrograms as .npy files |
| **Mel-Spectrogram Shape** | (1, 80, ~701 time steps) |
| **Total Size** | 5.41 GB (24.1k files) |

### **Data Collection Details**

- **Common Passage**: All speakers read similar text: *"Please call Stella..."*
- **Diverse Accents**: 60+ language backgrounds represented
- **Demographics**: Multiple age groups
- **Recording Quality**: Consistent recording environment (44.1kHz → 16kHz resampled)
- **Metadata**: Parquet file with timing, silence statistics, file paths

### **Preprocessing Pipeline**

Preprocessing and extensive EDA for our dataset creation documented in [notebook.ipynb](view) or [SpeechAccent_EDA.ipynb](open in google collab) or use direct link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link). This is part of our EDA that shows how our dataset is constructed. Dataset published on Kaggle for reproducibility:

1. **Audio Loading**: torchaudio library, mono conversion
2. **Silence Removal**: CodecSR-based voice activity detection
3. **Segmentation**: 7-second overlapping windows
4. **Mel-Spectrogram Generation**: 16kHz, N-FFT=400, hop=160, 80 mels
5. **dB Scale Conversion**: AmplitudeToDB transform
6. **Export**: Saved as .npy arrays with metadata

---

## 🔍 Exploratory Data Analysis

**Google Collab Notebook**: Preprocessing and extensive EDA for our dataset creation documented in [notebook.ipynb](view) or [SpeechAccent_EDA.ipynb](open in google collab) or use direct link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link)

### **Key Findings**

1. **Class Imbalance**: Non-English 77.2%, English 22.8% (3.4:1 ratio)
2. **Audio Quality**: Consistent recording environment, minimal noise
3. **Mel-Spectrogram Analysis**: Shape (1, 80, ~701), range -76 to +26 dB
4. **Feature Distributions**: Different spectral patterns between classes
5. **Data Quality**: No missing values, no corrupted files

### **EDA Steps Performed**

- ✅ Missing value analysis
- ✅ Duplicate detection  
- ✅ Class distribution analysis
- ✅ Mel-spectrogram shape verification
- ✅ Value range analysis
- ✅ Sample audio visualization
- ✅ Metadata completeness check

### **Feature Importance Analysis**

Our EDA continues in the **Training Kaggle Notebook**: [speechaccentkaggle.ipynb](https://www.kaggle.com/code/websaasai/speechaccentkaggle) on Kaggle (run out of GPU credits on Collab) with Feature Importance analysis.

**Analysis Files**: 
- [feature_importance_top30.csv](./kaggle-output/feature_importance_top30.csv)
- [separability_metrics.json](./kaggle-output/separability_metrics.json)

Before training deep learning models, we performed feature importance analysis using Random Forest on ResNet18-extracted features to assess class separability:

**Baseline Model Performance**:
- **Random Forest Accuracy**: 72.0%
- **ROC-AUC**: 0.796
- **CV Accuracy**: 71.0% (±1.4%)
- **Separation Ratio**: 1.013 (classes are distinguishable)

**Top 10 Most Important Features**:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | spectral_centroid_std | 0.0292 | Spectral Centroid |
| 2 | spectral_rolloff_std | 0.0248 | Spectral Rolloff |
| 3 | spectral_flatness_median | 0.0231 | Spectral Flatness |
| 4 | spectral_centroid_range | 0.0212 | Spectral Centroid |
| 5 | spectral_bandwidth_std | 0.0204 | Spectral Bandwidth |
| 6 | energy_delta_mean | 0.0197 | Energy Features |
| 7 | spectral_flatness_mean | 0.0183 | Spectral Flatness |
| 8 | spectral_std | 0.0182 | Spectral Statistics |
| 9 | energy_acceleration_mean | 0.0178 | Energy Features |
| 10 | formant_ratio | 0.0177 | Formant Features |

**Key Insights**:
- **Spectral features dominate**: Top features are spectral-based (centroid, rolloff, flatness)
- **72% baseline accuracy** indicates moderate class separability
- **Separation ratio 1.013**: Classes are distinguishable but challenging
- **Feature efficiency**: 51 features capture 80% importance, 60 features capture 90%
- **Conclusion**: CNN approach justified - automatic feature learning can outperform handcrafted features

This analysis validated our CNN approach, which achieved **+17.6% improvement** (72% → 89.6%).

---

## 🧠 Model Training

**Training Kaggle Notebook**: [speechaccentkaggle.ipynb](https://www.kaggle.com/code/websaasai/speechaccentkaggle) on Kaggle

### **Models Trained**

Multiple parameters and data augmentation strategieswere systematically tested across 3 models apart from the baseline Random Forest: 

**0. Baseline: Random Forest**
- Features: ResNet18-extracted (512-dim) + handcrafted (72 features)

**1. Model 1: ResNet18 Baseline (Conservative)**

**2. Model 2: ResNet18 Regularized (Balanced)**

**3. Model 3: ResNet18 Optimized (Aggressive)** ⭐ **BEST**

### **Hyperparameters Explored**

Multiple parameters were systematically tested across the 3 experiments:

- **Dropout rates**: [0.3, 0.5, 0.7] → Best: **0.7**
- **Learning rates**: [1e-4, 5e-5, 3e-5] → Best: **3e-5** with OneCycleLR
- **Batch sizes**: [32, 64] → Best: **64** (with heavy augmentation)
- **Weight decay**: [1e-4, 1e-3, 5e-4] → Best: **5e-4**
- **Optimizers**: [Adam, AdamW] → Best: **AdamW**
- **Schedulers**: [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR] → Best: **OneCycleLR**
- **SpecAugment intensity**: [None, 10%, 15%] → Best: **15%** (freq+time masking)
- **Mixup alpha**: [0.0, 0.2, 0.3] → Best: **0.3**
- **Layer freezing**: All conv layers frozen, FC trainable (feature extraction approach)
- **Early stopping patience**: [5, 7] epochs without improvement

### **Data Augmentation Strategies**

To combat overfitting (baseline 99.7% train → 72% test), we applied:

- **SpecAugment**: Frequency masking (15%) + time masking (15%)
- **Mixup**: Data mixing with α=0.3 for smoother decision boundaries
- **Label Smoothing**: 0.1 smoothing to prevent overconfident predictions
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

These augmentation strategies contributed to:
- **27.7% reduction in overfitting** (99.7% → 72% train-test gap)
- **17.6% accuracy improvement** over Random Forest baseline
- **Better generalization** with regularization (dropout 0.7)

### **Best Model Configuration**

- **Architecture**: ResNet18 (pre-trained ImageNet)
- **Dropout**: 0.7  
- **Frozen Layers**: All convolutional layers (11.2M params)
- **Trainable**: Final FC layer only (1,026 params)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Batch Size**: 32
- **Epochs**: 50 (early stopping)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss

**Key Finding**: Aggressive regularization (dropout 0.7) + heavy augmentation (SpecAugment 15%, Mixup 0.3) achieved best results.

**Training Script Exported**: [train.py](./train.py)

---

## 📈 Model Performance

### **Final Model Metrics**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 89.56% |
| **ROC-AUC** | 0.9382 |
| **English Precision** | 80.5% |
| **English Recall** | 71.5% |
| **Non-English Precision** | 91.8% |
| **Non-English Recall** | 94.9% |
| **F1-Score (English)** | 75.7% |
| **F1-Score (Non-English)** | 93.3% |

### **Confusion Matrix**

```
                    Predicted
                 Non-Eng  English
Actual Non-Eng   1,768      95      (94.9% recall)
       English     157     393      (71.5% recall)
```

### **Key Insights**

- **+17.6% improvement** over Random Forest baseline
- **Class imbalance impact**: Model more confident with Non-English
- **Calibration applied**: Threshold adjusted to 23% (from 50%)
- **Bayesian recalibration**: Adjusts for 3.4:1 training ratio

---

## 🚀 Quick Start

### **Option 1: Live Demo** (Fastest)

Visit [HuggingFace Space](https://huggingface.co/spaces/manthos/rate-your-english-accent)

### **Option 2: Docker**

```bash
docker pull mattkappa/rate-your-english-accent:latest
docker run -p 7860:7860 mattkappa/rate-your-english-accent:latest
# Access at http://localhost:7860
```

### **Option 3: Local Development**

```bash
git clone https://github.com/manthos/rate-your-english-accent.git
cd rate-your-english-accent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
# Access at http://localhost:7860
```

### **Option 4: Command-Line**

Build your local environment as in Option 3, then:
```bash
# Test with included samples from the dataset
python predict.py test-samples/english108.wav

# Or use your own audio file
python predict.py path/to/audio.wav
```

---

## 💻 Installation & Setup

### **Prerequisites**

- Python 3.10+
- pip or conda  
- 4GB+ RAM
- Optional: CUDA GPU

### **Installation Steps**

```bash
# Clone
git clone https://github.com/manthos/rate-your-english-accent.git
cd rate-your-english-accent

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (optional, for training)
kaggle datasets download -d websaasai/speech-accent
```

### **Model File**

Download `accent_classifier_traced.pt` (43MB):

1. From Kaggle output (if trained)
2. From GitHub Releases  
3. Train from scratch on gpu: `python train.py`

---

## 🔄 Reproducibility

### **Full Reproduction**

1. **Clone repo**: `git clone https://github.com/manthos/rate-your-english-accent.git`
2. **Download dataset**: From Kaggle
3. **Run EDA**: Open [notebook.ipynb](view) or [SpeechAccent_EDA.ipynb](open in google collab) or use direct link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link)
4. **Train model**: Run `speechaccentkaggle.ipynb` on Kaggle (GPU: T4 x2, ~2 hours)
5. **Test**: `python predict.py test-samples/english108.wav`
6. **Deploy**: `python app.py`

### **Reproducibility Checklist**

- ✅ Dataset with clear download instructions
- ✅ requirements.txt with pinned versions
- ✅ Random seeds set (random_state=42)
- ✅ Fully executable notebooks  
- ✅ Training script runs successfully
- ✅ Deterministic results (torch.manual_seed=42)

---

## 🌐 Model Deployment

### **Gradio Web Interface**

**File**: [app.py](./app.py)

**Features**:
- Audio upload (WAV/MP3/FLAC)
- Microphone recording
- Real-time prediction
- Dual probabilities (raw + calibrated)
- Confidence assessment

---

## 🐳 Containerization

### **Build & Run**

```bash
# Build
docker build -t rate-your-english-accent:latest .

# Run
docker run -p 7860:7860 rate-your-english-accent:latest

# Background
docker run -d -p 7860:7860 --name accent-classifier rate-your-english-accent:latest
```

### **Published to Docker Hub**

https://hub.docker.com/r/mattkappa/rate-your-english-accent

---

## ☁️ Cloud Deployment

### **HuggingFace Spaces**

**Live**: [https://huggingface.co/spaces/manthos/rate-your-english-accent](https://huggingface.co/spaces/manthos/rate-your-english-accent)

**Features**: 100% free, public URL, auto-HTTPS, Gradio auto-rendered

### **Alternative Platforms**

- Google Cloud Run
- AWS Elastic Beanstalk  
- Kubernetes

See [docs/deployment/](./deployment/) for guides.

---

## 📁 Project Structure

```
rate-your-english-accent/
├── README.md                          # This file
├── EVALUATION.md                      # Evaluation criteria
├── requirements.txt                   # Dependencies
├── Dockerfile                         # Docker config
├── app.py                             # Gradio interface
├── predict.py                         # CLI prediction
├── train.py                           # Training script
├── accent_classifier_traced.pt        # Trained model (43MB)
├── SpeechAccent_EDA.ipynb            # EDA notebook (save and open in Google Collab or use link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link)
├── notebook.ipynb                    # The above notebook but viewable on Github
├── speechaccentkaggle.ipynb          # Training notebook
├── test-samples/                      # Sample audio files some generated from Speech Accent Archive dataset
└── deployment/                        # Cloud guides
```

---

## 🛠️ Technology Stack

- **ML**: PyTorch 2.1.0, torchvision, torchaudio
- **Web**: Gradio 5.x
- **Deployment**: Docker, HuggingFace Spaces
- **Development**: Jupyter, Kaggle, Google Colab

See [requirements.txt](./requirements.txt) for full list.

---

## 📜 License

**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)

- **All Project Components**: CC BY-NC-SA 4.0
  - Source code
  - Trained model
  - Dataset and mel-spectrograms
  - Test samples
  - Documentation

**Attribution**:
- Original Dataset: Speech Accent Archive (George Mason University)
- Dataset Processing: WebSaasAi (CodecSR)
- Kaggle Dataset: https://www.kaggle.com/datasets/websaasai/speech-accent
- Test Samples: Audio files in `test-samples/` directory contains some generated from the Speech Accent Archive dataset, licensed under CC BY-NC-SA 4.0

---

## 📧 Contact

**Author**: manthos
- GitHub: [@manthos](https://github.com/manthos)
- Project: [rate-your-english-accent](https://github.com/manthos/rate-your-english-accent)

---

## 🙏 Acknowledgments

- ML Zoomcamp 2025 by DataTalks.Club
- Speech Accent Archive (GMU)
- WebSaasAi (our Kaggle dataset)
- PyTorch & Gradio communities
- HuggingFace (free hosting)

---

## 📚 Resources

- [Live Demo](https://huggingface.co/spaces/manthos/rate-your-english-accent)
- [Initial EDA Google Collab Notebook](Preprocessing and extensive EDA for our dataset creation documented in [notebook.ipynb](view) or [SpeechAccent_EDA.ipynb](open in google collab) or use direct link https://colab.research.google.com/drive/1RXLOrneNayAiu3zM34W047wP8a1DjRcS?usp=drive_link)
- [Further EDA, Feature Importance and Training Notebook](https://www.kaggle.com/code/websaasai/speechaccentkaggle)
- [Dataset](https://www.kaggle.com/datasets/websaasai/speech-accent)
- [Original Archive](https://accent.gmu.edu/)
- [Evaluation Criteria](./EVALUATION.md)

---

**⭐ If you found this project helpful, please star it on GitHub!**
