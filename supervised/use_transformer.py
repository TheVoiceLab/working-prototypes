import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import numpy as np
import os

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transformer_speech.pth")
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


# --- 2. MODEL DEFINITION ---
class SpeechTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Using the same bundle used during fine-tuning
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()

        # Classifier head (exactly as used in training)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x input shape: [Batch, Time]
        features, _ = self.feature_extractor(x)
        # Global Average Pooling (Mean) across the time dimension
        pooled = torch.mean(features, dim=1)
        return self.classifier(pooled)


# --- 3. PREDICTION LOGIC ---
def predict():
    # Load model
    model = SpeechTransformer(len(LABELS)).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Transformer model loaded from {MODEL_PATH}")
    else:
        print(f"❌ Error: {MODEL_PATH} not found. Please train the model first.")
        return

    print(f"\nListening for: {', '.join(LABELS)}")

    while True:
        input("\n[READY] Press Enter to record 1 second...")

        # Record 1 second of audio
        print("Recording...")
        rec = sd.rec(16000, samplerate=16000, channels=1)
        sd.wait()
        print("Processing...")

        # Prepare waveform
        waveform = rec.flatten()

        # 1. Peak Normalization (scales to -1.0 to 1.0)
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(DEVICE)

        # 2. Transformer-Specific Z-Score Normalization (Match Training)
        # This is critical for Wav2Vec self-attention layers
        waveform_tensor = (waveform_tensor - waveform_tensor.mean()) / (waveform_tensor.std() + 1e-7)

        # 3. Inference
        with torch.no_grad():
            output = model(waveform_tensor)
            probabilities = torch.softmax(output, dim=1)
            conf, idx = torch.max(probabilities, dim=1)

        result_label = LABELS[idx.item()]
        confidence = conf.item() * 100

        print("=" * 30)
        print(f"HEARD: {result_label.upper()}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("=" * 30)

        if input("Try again? (y/n): ").lower() == 'n':
            break


if __name__ == "__main__":
    predict()