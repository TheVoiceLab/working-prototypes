import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "speech_data")

target_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
class_to_idx = {label: i for i, label in enumerate(target_labels)}
idx_to_class = {v: k for k, v in class_to_idx.items()}


# --- 2. DATASET DEFINITION ---
class SpeechCommandsSubset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, download=True):
        super().__init__(root, download=download)
        self._walker = [fn for fn in self._walker if os.path.basename(os.path.dirname(fn)) in target_labels]

    def __getitem__(self, n):
        path = self._walker[n]
        label_idx = class_to_idx[os.path.basename(os.path.dirname(path))]
        try:
            speech, _ = sf.read(path)
            waveform = torch.from_numpy(speech).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            if waveform.size(1) < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
            else:
                waveform = waveform[:, :16000]
            return waveform, label_idx
        except:
            return torch.zeros((1, 16000)), label_idx


# --- 3. MODEL & TRANSFORMS ---
mel_transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512),
    T.AmplitudeToDB()
).to(device)


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


# --- 4. TRAINING & EVALUATION ---
def run_epoch(loader, model, optimizer, criterion, is_train=True):
    model.train() if is_train else model.eval()
    all_preds, all_targets = [], []

    for data, target in tqdm(loader, desc="Training" if is_train else "Validating"):
        data, target = data.to(device), target.to(device)
        spec = mel_transform(data)
        with torch.set_grad_enabled(is_train):
            output = model(spec)
            loss = criterion(output, target)
            if is_train:
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()

        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets


# --- 5. EXECUTION ---
if __name__ == "__main__":
    dataset = SpeechCommandsSubset(root=DATA_ROOT, download=True)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    model = SpeechCNN(len(target_labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 30):
        run_epoch(train_loader, model, optimizer, criterion, True)
        preds, targets = run_epoch(val_loader, model, optimizer, criterion, False)
        print(f"Completed Epoch {epoch}")

    # --- 6. VISUAL REFERENCE & METRICS ---
    print("\n" + "=" * 30)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 30)
    print(classification_report(targets, preds, target_names=target_labels))

    # Confusion Matrix Plot
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_labels, yticklabels=target_labels, cmap='Blues')
    plt.title('Speech Command Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # --- 7. EXAMPLES: INFERENCE TEST ---
    print("\n" + "=" * 30)
    print("MODEL INFERENCE EXAMPLES")
    print("=" * 30)
    model.eval()
    for i in range(5):
        # Pick a random sample from validation set
        idx = np.random.randint(len(val_set))
        waveform, true_label_idx = val_set[idx]

        # Prepare for model
        input_tensor = waveform.unsqueeze(0).to(device)
        spec = mel_transform(input_tensor)

        # Predict
        with torch.no_grad():
            output = model(spec)
            pred_idx = output.argmax(1).item()

        status = "✅ CORRECT" if pred_idx == true_label_idx else "❌ WRONG"
        print(
            f"Sample {i + 1}: Predicted: '{idx_to_class[pred_idx]}' | Actual: '{idx_to_class[true_label_idx]}' | {status}")

    torch.save(model.state_dict(), "speech_model.pth")
    print("\nProcess finished. Visuals saved.")