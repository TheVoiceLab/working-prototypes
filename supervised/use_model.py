import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
import os

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "speech_model.pth"
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds

target_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
idx_to_class = {i: label for i, label in enumerate(target_labels)}


# --- 2. MODEL DEFINITION (Must match training) ---
class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeechCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 8, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.fc_layers(self.conv_layers(x))


# --- 3. PREPARATION ---
# Initialize model and load weights
model = SpeechCNN(len(target_labels)).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print("❌ Error: speech_model.pth not found. Please train the model first.")
    exit()

# Match the MelSpectrogram settings from training
mel_transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512),
    T.AmplitudeToDB()
).to(device)


def predict_mic():
    print(f"\nReady! Say one of: {target_labels}")
    input("Press Enter to start recording for 1 second...")

    # Record audio
    print("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Processing...")

    # Convert to Tensor (Channel, Time)
    waveform = torch.from_numpy(recording.T).float().to(device)

    # Preprocess
    spec = mel_transform(waveform).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(spec)
        # Apply Softmax to get probabilities (optional)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    prediction = idx_to_class[pred_idx.item()]
    confidence = conf.item() * 100

    print("-" * 30)
    print(f"I HEARD: {prediction.upper()}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print("-" * 30)


# --- 4. RUN LOOP ---
if __name__ == "__main__":
    try:
        while True:
            predict_mic()
            cont = input("Try again? (y/n): ")
            if cont.lower() != 'y':
                break
    except KeyboardInterrupt:
        print("\nExiting...")