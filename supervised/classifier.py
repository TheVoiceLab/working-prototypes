import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# --- 1. SETUP PATHS AND DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use absolute path to avoid Windows/PyCharm relative path errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "speech_data")

if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

# Specific commands to classify
target_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
class_to_idx = {label: i for i, label in enumerate(target_labels)}


# --- 2. DATASET DEFINITION ---
class SpeechCommandsSubset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, download=True):
        super().__init__(root, download=download)
        # Filter the internal file list to only include our target labels
        self._walker = [fn for fn in self._walker if os.path.basename(os.path.dirname(fn)) in target_labels]

    def __getitem__(self, n):
        waveform, sample_rate, label, *_ = super().__getitem__(n)

        # Standardize to 16,000 samples (1 second)
        if waveform.size(1) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
        else:
            waveform = waveform[:, :16000]

        return waveform, class_to_idx[label]


# --- 3. PREPROCESSING PIPELINE ---
#
mel_transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512),
    T.AmplitudeToDB()
).to(device)


# --- 4. CNN ARCHITECTURE ---
class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeechCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


# --- 5. INITIALIZE DATA ---
full_dataset = SpeechCommandsSubset(root=DATA_ROOT, download=True)

# Simple 80/20 Train/Validation Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

model = SpeechCNN(len(target_labels)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# --- 6. TRAINING & EVALUATION LOOP ---
def run_epoch(loader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct = 0, 0

    for data, target in tqdm(loader, desc="Training" if is_train else "Validating"):
        data, target = data.to(device), target.to(device)
        spec = mel_transform(data)

        with torch.set_grad_enabled(is_train):
            output = model(spec)
            loss = criterion(output, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_acc = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), avg_acc


# Run for 5 Epochs
for epoch in range(1, 6):
    train_loss, train_acc = run_epoch(train_loader, is_train=True)
    val_loss, val_acc = run_epoch(val_loader, is_train=False)
    print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# Save the weights
torch.save(model.state_dict(), "speech_model.pth")