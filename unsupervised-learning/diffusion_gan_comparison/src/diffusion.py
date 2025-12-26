import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import numpy as np
import os

# ============================================================
# Conditional UNet
# ============================================================
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes, label_emd_dim, image_size):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emd_dim)
        self.image_size = image_size
        self.down1 = nn.Sequential(
            nn.Conv2d(1 + 1, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )
        self.out = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x, t, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        c = c.expand(-1, -1, self.image_size, self.image_size)

        t_emb = t.float().view(-1, 1, 1, 1)
        t_emb = t_emb.expand(-1, 1, self.image_size, self.image_size)

        x = torch.cat([x, t_emb], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.mid(d2)
        u1 = self.up1(m)
        out = self.out(u1)

        return out

class DiffusionDemo:
    def __init__(self):
        self.DATA_SIZE = 3000
        self.BATCH_SIZE = 64
        self.IMG_SIZE = 28
        self.NUM_CLASSES = 10
        self.LABEL_EMB_DIM = 32
        self.TIMESTEPS = 300
        self.LR = 1e-4
        self.EPOCHS = 100
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_DIR = '../output/diffusion_checkpoints'
        self.OUT_DIR = "../output/diffusion_generated_images"
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # ============================================================
        # Data Preparation, transform the images to avoid overfitting
        # ============================================================
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
        indices = np.random.choice(len(full_dataset), self.DATA_SIZE, replace=False)
        small_dataset = Subset(full_dataset, indices)
        loader = DataLoader(small_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.loader = loader
        # ============================================================
        # Noise Schedule, gradually add noises so that the generate can handle pure noise
        # ============================================================
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.TIMESTEPS).to(self.DEVICE)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.betas = betas
        alpha_hat = torch.cumprod(alphas, dim=0)
        self.alpha_hat = alpha_hat


    def save_model(self, model, epoch):
        path = f"{self.MODEL_DIR}/diffusion_epoch_{epoch}.pth"
        torch.save(model.state_dict(), path)
        print(f"✅ Saved model: {path}")

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        model.eval()
        print(f"✅ Loaded model: {path}")
        return model

    # ============================================================
    # Sampling
    # ============================================================
    @torch.no_grad()
    def sample_images(self, model, epoch):
        model.eval()
        n = 80
        labels = torch.arange(10).repeat(8).to(self.DEVICE)

        x = torch.randn(n, 1, self.IMG_SIZE, self.IMG_SIZE).to(self.DEVICE)

        for t in reversed(range(self.TIMESTEPS)):
            t_tensor = torch.full((n,), t, device=self.DEVICE)
            noise_pred = model(x, t_tensor, labels)

            alpha = self.alphas[t]
            alpha_hat_t = self.alpha_hat[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat_t)) * noise_pred) + torch.sqrt(beta) * noise

        vutils.save_image(x, f"{self.OUT_DIR}/epoch_{epoch}.png", nrow=10, normalize=True)
        print(f"✅ Saved samples for epoch {epoch}")
        model.train()

    def train(self):
        model = ConditionalUNet(num_classes=self.NUM_CLASSES,
                                label_emd_dim=self.LABEL_EMB_DIM,
                                image_size=self.IMG_SIZE).to(self.DEVICE)
        opt = optim.Adam(model.parameters(), lr=self.LR)

        for epoch in range(1, self.EPOCHS + 1):
            for real, labels in self.loader:
                real = real.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                b = real.size(0)
                t = torch.randint(0, self.TIMESTEPS, (b,), device=self.DEVICE)

                noise = torch.randn_like(real)
                alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)

                x_t = torch.sqrt(alpha_hat_t) * real + torch.sqrt(1 - alpha_hat_t) * noise

                noise_pred = model(x_t, t, labels)
                loss = nn.MSELoss()(noise_pred, noise)

                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f"Epoch {epoch}/{self.EPOCHS} | Loss: {loss.item():.4f}")

            if epoch % 10 == 0:
                self.sample_images(model, epoch)
                self.save_model(model, epoch)
        return model

    def generate(self, label):
        model = ConditionalUNet(
            num_classes=self.NUM_CLASSES,
            label_emd_dim=self.LABEL_EMB_DIM,
            image_size=self.IMG_SIZE
        ).to(self.DEVICE)
        ckpt = f"{self.MODEL_DIR}/diffusion_epoch_30.pth"

        if os.path.exists(ckpt):
            model = ConditionalUNet().to(self.DEVICE)
            model = self.load_model(model, ckpt)

        with torch.no_grad():
            labels = torch.full((64,), label, dtype=torch.long).to(self.DEVICE)
            x = torch.randn(64, 1, self.IMG_SIZE, self.IMG_SIZE).to(self.DEVICE)

            for t in reversed(range(proj.TIMESTEPS)):
                t_tensor = torch.full((64,), t, device=self.DEVICE)
                noise_pred = model(x, t_tensor, labels)

                alpha = self.alphas[t]
                alpha_hat_t = self.alpha_hat[t]
                beta = self.betas[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_hat_t)) * noise_pred) + torch.sqrt(beta) * noise

            vutils.save_image(x, f"test_generated_{label}.png", normalize=True)
            print(f"✅ Generated digit '{label}' from loaded model")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    proj = DiffusionDemo()
    model = proj.train()
    print("✅ Training complete.")

    # Example: Load a checkpoint and generate digits "7"
    proj.generate(8)