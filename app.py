import gradio as gr
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION (Match training parameters)
# ============================================================================
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
SAMPLE_RATE = 16000
TARGET_SIZE = (224, 224)

# ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Load model
MODEL_PATH = "accent_classifier_traced.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model on {device}...")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print("✅ Model loaded!")

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def preprocess_audio(audio_path):
    """Convert audio to model-ready tensor"""
    
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz
    if orig_sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Generate mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec = mel_transform(waveform)
    
    # Convert to dB
    db_transform = T.AmplitudeToDB()
    mel_spec_db = db_transform(mel_spec)
    
    # DO NOT normalize to 0-1 here! Training data was in raw dB scale (~-80 to +26 dB)
    # The ImageNet normalization will handle scaling
    
    # Handle shape
    if mel_spec_db.shape[0] == 1:
        mel_spec_db = mel_spec_db.squeeze(0)
    
    if len(mel_spec_db.shape) == 2:
        mel_spec_db = mel_spec_db.unsqueeze(0)
    
    # Replicate to 3 channels
    mel_spec_db = mel_spec_db.repeat(3, 1, 1)
    
    # Resize to 224x224
    mel_spec_db = mel_spec_db.unsqueeze(0)
    mel_spec_resized = torch.nn.functional.interpolate(
        mel_spec_db, size=TARGET_SIZE, mode='bilinear', align_corners=False
    )
    
    # Normalize with ImageNet stats
    for i in range(3):
        mel_spec_resized[0, i] = (mel_spec_resized[0, i] - NORM_MEAN[i]) / NORM_STD[i]
    
    return mel_spec_resized.to(device)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_accent(audio):
    """Main prediction function for Gradio"""
    
    if audio is None:
        return "⚠️ Please upload an audio file", None
    
    try:
        # Preprocess
        audio_tensor = preprocess_audio(audio)
        
        # Predict
        with torch.no_grad():
            logits = model(audio_tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Get raw probabilities (biased by training distribution)
        non_english_prob_raw = probs[0, 0].item()
        english_prob_raw = probs[0, 1].item()
        
        # Calibrate probabilities to balanced dataset (for display only)
        # Training ratio: 3.4:1 (Non-English:English) → prior=0.227
        # Balanced ratio: 1:1 → prior=0.5
        # Adjustment factor: 0.5 / 0.227 ≈ 2.2 for English, 0.5 / 0.773 ≈ 0.65 for Non-English
        TRAIN_PRIOR_ENGLISH = 0.227
        TRAIN_PRIOR_NON_ENGLISH = 0.773
        BALANCED_PRIOR = 0.5
        
        english_adjusted = english_prob_raw * (BALANCED_PRIOR / TRAIN_PRIOR_ENGLISH)
        non_english_adjusted = non_english_prob_raw * (BALANCED_PRIOR / TRAIN_PRIOR_NON_ENGLISH)
        
        # Renormalize to sum to 1
        total = english_adjusted + non_english_adjusted
        english_prob_calibrated = english_adjusted / total
        non_english_prob_calibrated = non_english_adjusted / total
        
        # Prediction (using raw probabilities with calibrated threshold)
        # Threshold = 1/(1+3.4) ≈ 0.23 compensates for training distribution
        THRESHOLD = 0.23  # Calibrated to 3.4:1 training ratio
        predicted_class = "English" if english_prob_raw > THRESHOLD else "Non-English"
        
        # Use calibrated probability for confidence assessment (more intuitive for users)
        confidence_calibrated = max(english_prob_calibrated, non_english_prob_calibrated)
        
        # Format output
        result = f"## 🎤 Prediction: **{predicted_class}**\n\n"
        
        result += f"### Model Output (Raw Probabilities):\n"
        result += f"- English: **{english_prob_raw:.1%}** (threshold: 23%)\n"
        result += f"- Non-English: **{non_english_prob_raw:.1%}**\n\n"
        
        result += f"### Calibrated for Balanced Dataset:\n"
        result += f"- English: **{english_prob_calibrated:.1%}**\n"
        result += f"- Non-English: **{non_english_prob_calibrated:.1%}**\n"
        result += f"*(Adjusted to compensate for 3.4:1 training imbalance)*\n\n"
        
        # Confidence assessment (using calibrated probabilities for intuitive interpretation)
        if 0.20 <= english_prob_raw <= 0.40:
            result += "⚠️ **BORDERLINE** - Close to decision boundary (23% threshold)\n"
            result += "💡 Audio characteristics may differ from training data sources\n"
        elif confidence_calibrated > 0.85:
            result += "✅ **HIGH CONFIDENCE** - Very clear accent characteristics"
        elif confidence_calibrated > 0.65:
            result += "⚠️ **MEDIUM CONFIDENCE** - Detectable but not strong"
        else:
            result += "⚠️ **LOW CONFIDENCE** - Near decision boundary"
        
        # Probability chart data (using calibrated probabilities for visualization)
        prob_dict = {
            "Raw - Non-English": non_english_prob_raw,
            "Raw - English": english_prob_raw,
            "Calibrated - Non-English": non_english_prob_calibrated,
            "Calibrated - English": english_prob_calibrated
        }
        
        return result, prob_dict
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
title = "🎙️ Rate Your English Accent"
description = """
### Classify speech as **English** or **Non-English** accent

Upload an audio file (.wav, .mp3, .flac) and the model will predict whether the speaker has an English accent.

**English accents:**
- American and a couple of Irish

**⚠️ Important Notes:**
- **Small class project** - trained on specific dataset (Speech Accent Archive)
- **Best performance**: All training samples used the same reading passage (see below)
- **For optimal results, read this passage:**

> *"Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."*

- **Audio quality**: Use similar recording environment for best results (check test-samples)
- Different microphones, codecs, or background noise may affect accuracy
- Model shows two probability sets: raw (threshold 23%) and calibrated (balanced)

**Training Details:**
- Model: ResNet18 CNN on mel-spectrograms
- Dataset: Speech Accent Archive processed with CodecSR (2,413 test samples, 3.4:1 class imbalance)
- Processing: 7-second segments with silence removal, 16kHz, 80 mel-spectrograms
- Test Accuracy: 89.6% | English Recall: 71.5% | Non-English Recall: 94.9%
- Data Source: [Speech Accent Split and Mel-spectrograms](https://www.kaggle.com/datasets/websaasai/speech-accent) (CC BY-NC-SA 4.0)
"""

article = """
### 📊 How It Works

1. **Audio Processing**: Converts audio to mel-spectrogram (80 mel-frequency bins, 16kHz)
2. **CNN Classification**: ResNet18 pre-trained on ImageNet, fine-tuned for accent classification
3. **Binary Prediction**: English vs Non-English accent

### 🎯 Model Details
- **Architecture**: ResNet18 with dropout regularization (0.7)
- **Input**: 224x224 mel-spectrograms (80 mels, 16kHz)
- **Training**: Class imbalance 3.4:1 (Non-English:English)
- **Performance**: 89.6% accuracy, 0.94 ROC-AUC, 23% decision threshold

### ⚠️ Limitations
- Trained on Speech Accent Archive processed with CodecSR - different audio sources may show reduced accuracy
- Class imbalance (3.4:1) means model is more confident with Non-English predictions
- Works best with similar recording quality and conditions as training data (44.1kHz → 16kHz, 7-second segments)
- May struggle with mixed accents, code-switching, or low-quality recordings

### 📚 Dataset Attribution
**Original Source:** Speech Accent Archive (George Mason University)  
**Processed Dataset:** [Speech Accent Split and Mel-spectrograms](https://www.kaggle.com/datasets/websaasai/speech-accent) by WebSaasAi  
**Processing:** CodecSR-based silence removal, 7-second overlap-split, mel-spectrogram generation  
**License:** CC BY-NC-SA 4.0

### 🔗 Links
- [GitHub Repository](https://github.com/manthos/rate-your-english-accent)
- [Processed Dataset on Kaggle](https://www.kaggle.com/datasets/websaasai/speech-accent)
- [Training Notebook](https://www.kaggle.com/code/websaasai/speechaccentkaggle)
- [Original Speech Accent Archive](https://accent.gmu.edu/)
"""

# Create interface
demo = gr.Interface(
    fn=predict_accent,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.JSON(label="Probability Scores")
    ],
    title=title,
    description=description,
    article=article
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
