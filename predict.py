#!/usr/bin/env python3
"""
predict.py - Production Inference Script for Rate Your English Accent
Classifies audio files as English or Non-English accent

VERIFIED PARAMETERS (from training pipeline):
- n_mels: 80
- n_fft: 400
- hop_length: 160
- sample_rate: 16000

Usage:
    python predict.py --audio path/to/audio.wav
    python predict.py --audio audio.mp3 --threshold 0.7
    python predict.py --audio recording.flac --device cuda
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import argparse
import sys
from pathlib import Path

# ============================================================================
# 1. Define global constants for mel-spectrogram generation
# ============================================================================
N_MELS = 80           # Number of mel-frequency bins
N_FFT = 400             # FFT window size
HOP_LENGTH = 160   # Number of samples between frames
SAMPLE_RATE = 16000 # Target sample rate (Hz)

# Model expects 224x224 input
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# ImageNet normalization (used during training)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Class mapping
CLASS_NAMES = {0: 'Non-English', 1: 'English'}

# Default model path
DEFAULT_MODEL_PATH = 'accent_classifier_traced.pt'

# ============================================================================
# 2. Preprocessing Function
# ============================================================================
def preprocess_audio(audio_path: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load and preprocess audio file into model-ready tensor.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
        device: 'cpu' or 'cuda'
    
    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224)
    """
    print(f"\n🔊 Loading audio: {audio_path}")
    
    # 1. Load audio
    try:
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        print(f"   Original: {orig_sample_rate} Hz, {waveform.shape[1]} samples")
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    
    # 2. Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"   Converted to mono")
    
    # 3. Resample to target sample rate
    if orig_sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        print(f"   Resampled: {orig_sample_rate} Hz → {SAMPLE_RATE} Hz")
    
    # 4. Generate mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec = mel_transform(waveform)
    print(f"   Mel-spectrogram: {mel_spec.shape}")
    
    # 5. Convert to dB scale
    db_transform = T.AmplitudeToDB()
    mel_spec_db = db_transform(mel_spec)
    
    # DO NOT normalize to 0-1 here! Training data was in raw dB scale (~-80 to +26 dB)
    # The ImageNet normalization will handle scaling
    
    # 6. Handle shape: (1, n_mels, time) → (n_mels, time)
    if mel_spec_db.shape[0] == 1:
        mel_spec_db = mel_spec_db.squeeze(0)
    
    # 7. Add channel dimension if needed
    if len(mel_spec_db.shape) == 2:
        mel_spec_db = mel_spec_db.unsqueeze(0)
    
    # 8. Replicate to 3 channels (RGB) for ResNet18
    mel_spec_db = mel_spec_db.repeat(3, 1, 1)  # (3, n_mels, time)
    print(f"   Replicated to 3 channels: {mel_spec_db.shape}")
    
    # 9. Resize to 224x224 (bilinear interpolation)
    mel_spec_db = mel_spec_db.unsqueeze(0)  # Add batch dim: (1, 3, n_mels, time)
    mel_spec_resized = torch.nn.functional.interpolate(
        mel_spec_db,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode='bilinear',
        align_corners=False
    )
    print(f"   Resized: {mel_spec_resized.shape}")
    
    # 10. Normalize using ImageNet stats
    for i in range(3):
        mel_spec_resized[0, i] = (mel_spec_resized[0, i] - NORM_MEAN[i]) / NORM_STD[i]
    
    # 11. Move to device
    mel_spec_final = mel_spec_resized.to(device)
    
    print(f"   ✅ Final shape: {mel_spec_final.shape}")
    return mel_spec_final

# ============================================================================
# 3. Inference Function
# ============================================================================
def predict(model, audio_tensor: torch.Tensor, threshold: float = 0.23) -> dict:
    """
    Run inference on preprocessed audio tensor.
    
    Args:
        model: Loaded PyTorch model
        audio_tensor: Preprocessed tensor (1, 3, 224, 224)
        threshold: Classification threshold (default 0.23 for 3.4:1 class imbalance)
    
    Returns:
        Dictionary with prediction results including raw and calibrated probabilities
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits = model(audio_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get raw probabilities (biased by 3.4:1 training distribution)
        non_english_prob_raw = probs[0, 0].item()
        english_prob_raw = probs[0, 1].item()
        
        # Calibrate probabilities to balanced dataset
        # Training ratio: 3.4:1 (Non-English:English) → prior=0.227
        # Balanced ratio: 1:1 → prior=0.5
        TRAIN_PRIOR_ENGLISH = 0.227
        TRAIN_PRIOR_NON_ENGLISH = 0.773
        BALANCED_PRIOR = 0.5
        
        english_adjusted = english_prob_raw * (BALANCED_PRIOR / TRAIN_PRIOR_ENGLISH)
        non_english_adjusted = non_english_prob_raw * (BALANCED_PRIOR / TRAIN_PRIOR_NON_ENGLISH)
        
        # Renormalize to sum to 1
        total = english_adjusted + non_english_adjusted
        english_prob_calibrated = english_adjusted / total
        non_english_prob_calibrated = non_english_adjusted / total
        
        # Apply threshold to raw probability
        final_class = 1 if english_prob_raw >= threshold else 0
        
        return {
            'predicted_class': final_class,
            'predicted_label': CLASS_NAMES[final_class],
            'raw_probabilities': {
                'english': english_prob_raw,
                'non_english': non_english_prob_raw
            },
            'calibrated_probabilities': {
                'english': english_prob_calibrated,
                'non_english': non_english_prob_calibrated
            },
            'threshold': threshold,
            'raw_logits': logits[0].cpu().numpy().tolist()
        }

# ============================================================================
# 4. Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Classify audio as English or Non-English accent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --audio sample.wav
    python predict.py --audio voice.mp3 --threshold 0.7
    python predict.py --audio recording.flac --device cuda
    python predict.py --audio sample.wav --model custom_model.pt
        """
    )
    
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to TorchScript model (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--threshold', type=float, default=0.23,
                        help='Classification threshold (default: 0.23 for 3.4:1 class imbalance)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.audio).exists():
        print(f"❌ Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print("="*80)
    print("🎯 ACCENT CLASSIFICATION")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Audio: {args.audio}")
    print(f"   Model: {args.model}")
    print(f"   Device: {device}")
    print(f"   Threshold: {args.threshold}")
    
    # Load model
    print(f"\n🔧 Loading model...")
    try:
        model = torch.jit.load(args.model, map_location=device)
        print(f"   ✅ Loaded TorchScript model from {args.model}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        print(f"   Make sure the model file is a TorchScript traced model (.pt)")
        sys.exit(1)
    
    # Preprocess audio
    try:
        audio_tensor = preprocess_audio(args.audio, device)
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        sys.exit(1)
    
    # Run inference
    print(f"\n🤖 Running inference...")
    try:
        results = predict(model, audio_tensor, args.threshold)
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        sys.exit(1)
    
    # Display results
    print(f"\n" + "="*80)
    print("📊 PREDICTION RESULTS")
    print("="*80)
    print(f"\n🎤 Predicted Accent: {results['predicted_label']}")
    
    print(f"\n📈 Raw Probabilities (Model Output):")
    print(f"   English:     {results['raw_probabilities']['english']:.2%} (threshold: {args.threshold:.0%})")
    print(f"   Non-English: {results['raw_probabilities']['non_english']:.2%}")
    
    print(f"\n📊 Calibrated Probabilities (Balanced Dataset Equivalent):")
    print(f"   English:     {results['calibrated_probabilities']['english']:.2%}")
    print(f"   Non-English: {results['calibrated_probabilities']['non_english']:.2%}")
    print(f"   (Adjusted to compensate for 3.4:1 training imbalance)")
    
    print(f"\n🎯 Decision:")
    print(f"   Threshold: {args.threshold} (calibrated for 3.4:1 class imbalance)")
    print(f"   Final Classification: {results['predicted_label']}")
    
    # Confidence interpretation
    conf_raw = results['raw_probabilities']['english']
    if 0.20 <= conf_raw <= 0.40:
        print(f"   Confidence Level: BORDERLINE ⚠️ (close to decision boundary)")
    elif conf_raw > 0.85 or conf_raw < 0.15:
        print(f"   Confidence Level: HIGH ✅")
    elif conf_raw > 0.65 or conf_raw < 0.35:
        print(f"   Confidence Level: MEDIUM ⚠️")
    else:
        print(f"   Confidence Level: LOW ⚠️")
    
    print(f"\n✅ Inference complete!")
    
    return results

if __name__ == '__main__':
    main()
