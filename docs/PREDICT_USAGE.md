# predict.py Usage Guide

## 🚀 Quick Start

```bash
python predict.py --audio sample.wav --model accent_classifier_traced.pt
```

## 📋 Requirements

Install dependencies:
```bash
pip install torch torchaudio numpy
```

## 🎯 Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--audio` | ✅ Yes | - | Path to input audio file (.wav, .mp3, .flac) |
| `--model` | ✅ Yes | - | Path to trained model (accent_classifier_traced.pt) |
| `--threshold` | ❌ No | 0.5 | Classification threshold (0.0-1.0) |
| `--device` | ❌ No | cpu | Device: 'cpu' or 'cuda' |

## 📊 Examples

### Basic usage
```bash
python predict.py --audio recording.wav --model accent_classifier_traced.pt
```

### With custom threshold (more sensitive to English)
```bash
python predict.py --audio voice.mp3 --model accent_classifier_traced.pt --threshold 0.3
```

### GPU inference
```bash
python predict.py --audio audio.flac --model accent_classifier_traced.pt --device cuda
```

## 🔧 Preprocessing Pipeline

The script automatically handles:

1. ✅ **Audio Loading** - Supports .wav, .mp3, .flac, etc.
2. ✅ **Resampling** - Converts to 16000 Hz
3. ✅ **Mono Conversion** - Stereo → mono averaging
4. ✅ **Mel-Spectrogram** - Using verified parameters:
   - `n_mels=80`
   - `n_fft=400`
   - `hop_length=160`
5. ✅ **dB Conversion** - Amplitude to decibel scale
6. ✅ **Normalization** - ImageNet mean/std
7. ✅ **Resizing** - To 224x224 for ResNet18
8. ✅ **Channel Replication** - 1 channel → 3 channels (RGB)

## 📈 Output Format

```
================================================================================
📊 PREDICTION RESULTS
================================================================================

🎤 Predicted Accent: English

📈 Confidence Scores:
   English:     87.34%
   Non-English: 12.66%

🎯 Decision:
   Threshold: 0.5
   Final Classification: English
   Confidence Level: HIGH ✅

✅ Inference complete!
```

## 🎚️ Threshold Tuning

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| **0.3** | More sensitive to English | Minimize false negatives (catch more English) |
| **0.5** | Balanced (default) | Equal treatment of both classes |
| **0.7** | More conservative | Minimize false positives (only confident English) |

## ⚠️ Common Issues

### Issue: "CUDA not available"
**Solution:** Install PyTorch with CUDA support or use `--device cpu`

### Issue: "Audio file not found"
**Solution:** Check file path, use absolute path if needed

### Issue: "Model file not found"
**Solution:** Ensure `accent_classifier_traced.pt` is in the same directory

### Issue: "Failed to load audio file"
**Solution:** Check audio format, try converting to .wav using:
```bash
ffmpeg -i input.mp3 output.wav
```

## 🧪 Testing

Test with sample audio:
```bash
python test_predict.py
```

## 📦 Deployment

For production deployment:
1. Use TorchScript model (`accent_classifier_traced.pt`)
2. Deploy with Docker/Flask/FastAPI
3. See `deployment_guide.md` for details

## 🔍 Debugging

Enable verbose output:
```python
# In predict.py, add after imports:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ✅ Verified Parameters

These match your training pipeline:
- Sample Rate: 16000 Hz
- N_MELS: 80
- N_FFT: 400
- Hop Length: 160

Source: `SpeechAccent_EDA.ipynb` (lines 629, 679-681)
