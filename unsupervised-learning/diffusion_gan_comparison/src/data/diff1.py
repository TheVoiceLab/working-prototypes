import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
import os


# --- 1. Conditional U-Net ---
class ConditionalTinyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Time Embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        # Class Embedding (10 digits, 128-dim vector each)
        self.label_embed = nn.Embedding(10, 128)

        self.down1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)

        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv2d(64 + 64, 1, kernel_size=3, padding=1)

    def forward(self, x, t, labels):
        # Embed time and labels
        t_emb = self.t_embed(t.float().view(-1, 1)).view(-1, 128, 1, 1)
        l_emb = self.label_embed(labels).view(-1, 128, 1, 1)

        # Combine embeddings (Conditioning)
        condition = t_emb + l_emb

        h1 = F.silu(self.norm1(self.down1(x)))
        h2 = F.silu(self.norm2(self.down2(h1)))

        # Inject condition into bottleneck
        h_mid = self.mid(h2 + condition)

        h_up = F.silu(self.up1(h_mid))
        out = self.final(torch.cat([h_up, h1], dim=1))
        return out


# --- 2. Diffusion Manager ---
class Diffusion:
    def __init__(self, steps=500, device="cuda"):
        self.steps = steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def noise_image(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise

    @torch.no_grad()
    def sample(self, model, n, labels):
        model.eval()
        x = torch.randn((n, 1, 28, 28)).to(self.device)
        for i in reversed(range(self.steps)):
            t = torch.full((n,), i, dtype=torch.long).to(self.device)
            predicted_noise = model(x, t, labels)  # Pass labels during sampling

            alpha = self.alpha[t][:, None, None, None]
            alpha_cumprod = self.alpha_cumprod[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            noise = torch.randn_like(x) if i > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise
            ) + torch.sqrt(beta) * noise
        model.train()
        return x


# --- 3. Training Script ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("output", exist_ok=True)

    epochs = 200
    batch_size = 128
    lr = 1e-3

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    dataset = Subset(full_dataset, list(range(30000)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalTinyUnet().to(device)
    diff = Diffusion(steps=500, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)

    print(f"Training on {device}...")
    for epoch in range(1, epochs + 1):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, diff.steps, (images.shape[0],)).to(device)

            x_noisy, noise = diff.noise_image(images, t)
            predicted_noise = model(x_noisy, t, labels)  # Pass labels here

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

        if epoch % 10 == 0 or epoch == 1:
            # Generate two rows of 0-7 digits to check conditioning
            sample_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]).to(device)
            samples = diff.sample(model, 16, sample_labels)
            samples = (samples.clamp(-1, 1) + 1) / 2
            grid = utils.make_grid(samples, nrow=8)
            utils.save_image(grid, f"output/c_epoch_{epoch}.png")

    print("Training Complete.")


if __name__ == "__main__":
    train()