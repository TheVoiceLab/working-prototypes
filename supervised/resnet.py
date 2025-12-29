import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchvision import models

# --- 0. BACKEND PATCH ---
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
try:
    import soundfile


    def forced_load(path, **kwargs):
        data, samplerate = soundfile.read(path)
        # Returns (Channel, Time)
        return torch.from_numpy(data).float().view(1, -1), samplerate


    torchaudio.load = forced_load
    print("✅ Successfully patched torchaudio to use soundfile.")
except ImportError:
    print("❌ ERROR: Please run 'pip install soundfile' in your terminal.")

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "speech_data")

BATCH_SIZE = 64
EPOCHS = 30
SAMPLE_RATE = 16000

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


# --- 2. DATASET CLASS ---
class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str):
        super().__init__(DATA_ROOT, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

        self._walker = [w for w in self._walker if any(f"{os.sep}{lab}{os.sep}" in w for lab in LABELS)]


def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        # Pad or trim to exactly 1 second
        if waveform.shape[1] > SAMPLE_RATE:
            waveform = waveform[:, :SAMPLE_RATE]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))

        tensors.append(waveform)
        targets.append(torch.tensor(LABEL_TO_IDX[label]))
    return torch.stack(tensors), torch.stack(targets)


# --- 3. THE IMPROVED RESNET ARCHITECTURE ---
def get_resnet_model(num_classes):
    # Use ResNet18 (Deep but lightweight)
    model = models.resnet18(weights=None)

    # FIX 1: Change initial conv from 7x7 (stride 2) to 3x3 (stride 1)
    # This prevents losing audio details in the first layer.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # FIX 2: Remove the MaxPool layer which is too aggressive for 64x32 spectrograms
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# --- 4. TRAINING LOOP ---
def train():
    # Audio transformation settings
    transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512),
        T.AmplitudeToDB()
    ).to(DEVICE)

    print("Loading datasets...")
    train_loader = DataLoader(SubsetSC("training"), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = get_resnet_model(len(LABELS))
    criterion = nn.CrossEntropyLoss()

    # Use AdamW for better weight decay handling in deep networks
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # FIX 3: Learning Rate Scheduler (Ramps up then cools down)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    print(f"Starting optimized ResNet training on {DEVICE}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Convert audio to Mel Spectrogram
            spec = transform(data)  # Output size: [Batch, 1, 64, 32]

            optimizer.zero_grad()
            output = model(spec)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            scheduler.step()  # Step every batch for OneCycleLR

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "resnet_speech_model.pth"))
    print(f"✅ Training complete. Model saved as resnet_speech_model.pth")


if __name__ == "__main__":
    train()