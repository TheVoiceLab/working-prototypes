import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import os
import numpy as np


class WassersteinGAN_Project:

    # ============================================================
    # 1. Config & Hyperparameters
    # ============================================================
    def __init__(self):
        self.DATA_SIZE = 3000
        self.BATCH_SIZE = 64
        self.Z_DIM = 128
        self.LABEL_EMB_DIM = 50
        self.IMG_SIZE = 28
        self.NUM_CLASSES = 10
        self.LR = 1e-4
        # WGAN-GP Parameters
        self.LAMBDA_GP = 10  # Weight for gradient penalty
        self.CRITIC_ITER = 5  # Train Discriminator 5 times for every 1 Generator step
        self.EPOCHS = 100  # More epochs for WGAN stability
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.OUT_DIR = "../output/wgan_generated_images"
        self.MODEL_DIR = "../output/wgan_model_checkpoints"

        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # ============================================================
    # 3. Gradient Penalty Function
    # ============================================================
    def compute_gradient_penalty(self, disc, real_samples, fake_samples, labels):
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(self.DEVICE)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = disc(interpolates, labels)
        fake = torch.ones(d_interpolates.shape).to(self.DEVICE)

        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    # ============================================================
    # 4. Training Loop
    # ============================================================
    def train(self):
        gen = (ProGenerator(num_classes=self.NUM_CLASSES, label_emb_dim=self.LABEL_EMB_DIM, noise_dim=self.Z_DIM)
               .to(self.DEVICE))
        disc = ProDiscriminator(num_class=self.NUM_CLASSES, image_size=self.IMG_SIZE).to(self.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=self.LR, betas=(0.0, 0.9))
        opt_disc = optim.Adam(disc.parameters(), lr=self.LR, betas=(0.0, 0.9))

        full_ds = datasets.MNIST(root="dataset/", train=True, download=True,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        indices = np.random.choice(len(full_ds), self.DATA_SIZE, replace=False)
        loader = DataLoader(Subset(full_ds, indices), batch_size=self.BATCH_SIZE, shuffle=True)

        for epoch in range(1, self.EPOCHS + 1):
            for i, (real, labels) in enumerate(loader):
                real, labels = real.to(self.DEVICE), labels.to(self.DEVICE)

                # --- Train Critic (Discriminator) ---
                for _ in range(self.CRITIC_ITER):
                    noise = torch.randn(real.size(0), self.Z_DIM, 1, 1).to(self.DEVICE)
                    fake = gen(noise, labels)

                    # WGAN Loss: maximize (D(real) - D(fake)) -> minimize (D(fake) - D(real))
                    d_real = disc(real, labels)
                    d_fake = disc(fake.detach(), labels)
                    gp = self.compute_gradient_penalty(disc, real.data, fake.data, labels)

                    loss_d = torch.mean(d_fake) - torch.mean(d_real) + self.LAMBDA_GP * gp
                    disc.zero_grad();
                    loss_d.backward();
                    opt_disc.step()

                # --- Train Generator ---
                noise = torch.randn(real.size(0), self.Z_DIM, 1, 1).to(self.DEVICE)
                fake = gen(noise, labels)
                # Generator wants to maximize Critic's score for fake images
                loss_g = -torch.mean(disc(fake, labels))
                gen.zero_grad();
                loss_g.backward();
                opt_gen.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.EPOCHS} | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")
                self.save_samples(gen, epoch)
                torch.save(gen.state_dict(), f"{self.MODEL_DIR}/pro_gen_{epoch}.pth")

    def save_samples(self, gen, epoch):
        gen.eval()
        with torch.no_grad():
            labels = torch.arange(10).repeat(8).to(self.DEVICE)
            noise = torch.randn(80, self.Z_DIM, 1, 1).to(self.DEVICE)
            samples = gen(noise, labels)
            vutils.save_image(samples, f"{self.OUT_DIR}/pro_epoch_{epoch}.png", nrow=10, normalize=True)
        gen.train()

class ProGenerator(nn.Module):
    def __init__(self,num_classes, label_emb_dim, noise_dim):
        super().__init__()
        NUM_CLASSES = num_classes
        LABEL_EMB_DIM = label_emb_dim
        Z_DIM = noise_dim
        self.label_emb = nn.Embedding(NUM_CLASSES, LABEL_EMB_DIM)
        #The projection layer blends noise with label
        self.l1 = nn.Sequential(
            nn.Linear(Z_DIM + LABEL_EMB_DIM, 256 * 7 * 7),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            #Conv2d increase channels to 128*4, then pixshuffle reduct channel by 4
            #and then increase width and height by 2
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 14x14 -> 28x28
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Final Layer
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        z = torch.cat((noise.view(noise.size(0), -1), c), dim=1)
        x = self.l1(z).view(-1, 256, 7, 7)
        return self.conv_blocks(x)


class ProDiscriminator(nn.Module):
    def __init__(self, num_class, image_size):
        super().__init__()
        NUM_CLASSES = num_class
        IMG_SIZE = image_size
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)

        def sn_conv(in_f, out_f, stride=2):
            return nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_f, out_f, 3, stride, 1)
            )

        self.model = nn.Sequential(
            sn_conv(2, 64), #28 * 28 -> 14 * 14, as defaut stride is 2
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(64, 128), #14 * 14 -> 7 * 7
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(128, 256), # 7 * 7 -> 4 * 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1)  # Output is a score, no Sigmoid for WGAN
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, 28, 28)
        x = torch.cat([img, c], dim=1) # input shape 2 * 28 * 28
        return self.model(x)


if __name__ == "__main__":
    project = WassersteinGAN_Project()
    project.train()