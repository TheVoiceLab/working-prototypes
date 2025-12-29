import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
import os
from torchvision import models

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the NEW optimized model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet_speech_model.pth")

SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second window

TARGET_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
IDX_TO_CLASS = {i: label for i, label in enumerate(TARGET_LABELS)}


# --- 2. UPDATED MODEL LOADER ---
def load_trained_model(path, num_classes):
    # Recreate the OPTIMIZED ResNet18 architecture
    model = models.resnet18(weights=None)

    # FIX 1: Change initial conv to 3x3 (stride 1) to match training
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # FIX 2: Disable MaxPool to match training
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if os.path.exists(path):
        # Load weights into the modified architecture
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Loaded OPTIMIZED weights from {path}")
        return model
    else:
        raise FileNotFoundError(f"❌ Error: {path} not found. Ensure you trained with the 3x3 kernel version.")


# --- 3. PRE-PROCESSING PIPELINE ---
mel_transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512),
    T.AmplitudeToDB()
).to(DEVICE)


def predict_audio(model):
    print(f"\n--- Ready! Say one of: {TARGET_LABELS} ---")
    input("Press ENTER to start 1-second recording...")

    # Record from mic
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    # --- NORMALIZATION ---
    audio = audio.flatten()
    # Peak normalization
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Convert to Tensor [Batch=1, Channel=1, Time]
    waveform = torch.from_numpy(audio).unsqueeze(0).float().to(DEVICE)

    # Transform to MelSpectrogram [Batch=1, Channel=1, Mels, Time]
    spec = mel_transform(waveform).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(spec)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    label = IDX_TO_CLASS[pred_idx.item()]
    conf_score = confidence.item() * 100

    print("-" * 30)
    print(f"RESULT: {label.upper()}")
    print(f"CONFIDENCE: {conf_score:.2f}%")
    print("-" * 30)


# --- 4. MAIN RUNNER ---
if __name__ == "__main__":
    try:
        # Check if model exists before starting
        speech_model = load_trained_model(MODEL_PATH, len(TARGET_LABELS))

        while True:
            predict_audio(speech_model)
            choice = input("\nTry another word? (y/n): ").lower()
            if choice != 'y':
                break
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("Exiting inference script.")