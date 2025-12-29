import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

# FORCE BACKEND FIX
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
try:
    import soundfile

    torchaudio.load = lambda path: (torch.from_numpy(soundfile.read(path)[0]).float().view(1, -1), 16000)
except ImportError:
    print("Please install soundfile: pip install soundfile")

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "speech_data")

# Transformers are VRAM hungry; keep Batch Size small
BATCH_SIZE = 8
EPOCHS = 20
# FIX: Transformers need a MUCH smaller Learning Rate for fine-tuning
LR = 1e-5

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


# --- 2. THE TRANSFORMER MODEL ---
class SpeechTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()

        # FIX: UNFREEZE the encoder. This is why your accuracy was poor.
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Wav2Vec expects (Batch, Time)
        features, _ = self.feature_extractor(x)
        # Pooling across the time dimension
        x = torch.mean(features, dim=1)
        return self.classifier(x)


# --- 3. DATA LOADING ---
class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str):
        super().__init__(DATA_ROOT, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        if subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]
        self._walker = [w for w in self._walker if any(f"{os.sep}{l}{os.sep}" in w for l in LABELS)]


def collate_fn(batch):
    tensors = []
    targets = []
    for b in batch:
        waveform = b[0]
        # Pad/Trim to exactly 16000 samples
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        # FIX: Normalization is essential for Transformer attention layers
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

        tensors.append(waveform.squeeze(0))
        targets.append(torch.tensor(LABEL_TO_IDX[b[2]]))

    return torch.stack(tensors), torch.stack(targets)


# --- 4. TRAIN ---
def train():
    train_loader = DataLoader(SubsetSC("training"), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = SpeechTransformer(len(LABELS)).to(DEVICE)

    # Use AdamW (better for transformers) and a Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"Fine-Tuning Transformer on {DEVICE}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        scheduler.step()
        print(f"Epoch {epoch} - Loss: {loss_total / len(train_loader):.4f} - LR: {optimizer.param_groups[0]['lr']:.7f}")

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "transformer_speech.pth"))
    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    train()