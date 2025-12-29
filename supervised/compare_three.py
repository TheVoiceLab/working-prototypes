import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import soundfile as sf

# --- 0. ENV SETUP ---
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "speech_data")
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


# --- 1. ARCHITECTURE DEFINITIONS (Must match training exactly) ---

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


def get_resnet_optimized(num_classes):
    model = models.resnet18(weights=None)
    # Match the optimized training: 3x3 kernel, stride 1, and Identity maxpool
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SpeechTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat, _ = self.feature_extractor(x)
        return self.classifier(torch.mean(feat, dim=1))


# --- 2. DATA LOADER ---
class TestDataset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root):
        super().__init__(root, download=True, subset="testing")
        self._walker = [fn for fn in self._walker if os.path.basename(os.path.dirname(fn)) in LABELS]

    def __getitem__(self, n):
        path = self._walker[n]
        label = os.path.basename(os.path.dirname(path))
        speech, _ = sf.read(path)
        waveform = torch.from_numpy(speech).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        if waveform.size(1) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
        else:
            waveform = waveform[:, :16000]
        return waveform, LABEL_TO_IDX[label]


# --- 3. EVALUATION LOGIC ---
def evaluate(model, loader, name):
    model.eval()
    all_preds, all_targets = [], []

    # Same transforms as ResNet/CNN training
    mel_tf = nn.Sequential(
        T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512),
        T.AmplitudeToDB()
    ).to(DEVICE)

    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            if name == "Transformer":
                # Transformer Normalization
                data_norm = (data - data.mean()) / (data.std() + 1e-7)
                output = model(data_norm.squeeze(1))
            else:
                # CNN/ResNet Spectrogram
                spec = mel_tf(data)
                output = model(spec)

            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.numpy())
    return all_targets, all_preds


if __name__ == "__main__":
    test_loader = DataLoader(TestDataset(DATA_ROOT), batch_size=32, shuffle=False)

    models_to_test = {
        "SpeechCNN": ("speech_model.pth", SpeechCNN(10)),
        "ResNet18_Opt": ("resnet_speech_model.pth", get_resnet_optimized(10)),
        "Transformer": ("transformer_speech.pth", SpeechTransformer(10))
    }

    final_results = []

    print("\n--- Starting Cross-Model Evaluation ---")

    for name, (path, model) in models_to_test.items():
        if os.path.exists(path):
            print(f"Testing {name}...")
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)

            y_true, y_pred = evaluate(model, test_loader, name)
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)

            final_results.append({
                "Model": name,
                "Accuracy": f"{acc * 100:.2f}%",
                "Avg Precision": f"{report['macro avg']['precision'] * 100:.2f}%",
                "Avg Recall": f"{report['macro avg']['recall'] * 100:.2f}%"
            })
        else:
            print(f"Skipping {name}: {path} not found.")

    # --- FINAL COMPARISON TABLE ---
    df = pd.DataFrame(final_results)
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (UNSEEN TEST DATA)")
    print("=" * 60)
    print(df.to_string(index=False))