import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import os
import numpy as np


class CGan_DigitGen:
    # ============================================================
    # Config & Hyperparameters
    # ============================================================
    def __init__(self):
        self.DATA_SIZE = 3000  # Number of MNIST samples to use
        self.BATCH_SIZE = 64  # Smaller batch for smaller data
        """
        We choose this ratio to avoid label dominance. When this happens,
        the digits generated on each row will be almost the same
        """
        self.Z_DIM = 200 # Latent Space Dimension (Noise Dimension)
        self.LABEL_EMB_DIM = 10 # Label Dimension

        self.NUM_CLASSES = 10
        self.IMG_SIZE = 28
        # TTUR: Generator learns faster than Discriminator
        self.G_LR = 0.0004
        self.D_LR = 0.0001
        self.EPOCHS = 100  # Small data needs more epochs to converge
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.OUT_DIR = "../output/cgan_generated_images"
        os.makedirs(self.OUT_DIR, exist_ok=True)
        self.MODEL_DIR = "../output/cgan_model_checkpoints"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
    # ============================================================
    # Data Preparation
    # ============================================================
    def data_preparation(self):
        transform = transforms.Compose([
            transforms.RandomRotation(10),  # turn it a little to avoid overfitting
            transforms.ToTensor(), # PIL image (0–255 integers) into a PyTorch tensor.
            transforms.Normalize((0.5,), (0.5,)) # center data around 0, range (-1,1)
            # Normalization ===> x = (x - 0.5) / 0.5
        ])
        self.raw_image_root = "./data/raw"
        full_dataset = datasets.MNIST(root=self.raw_image_root, train=True, transform=transform, download=True)
        # Create a small subset, 3000 images should be ok
        indices = np.random.choice(len(full_dataset), self.DATA_SIZE, replace=False)
        small_dataset = Subset(full_dataset, indices)
        self.loader = DataLoader(small_dataset, batch_size=self.BATCH_SIZE, shuffle=True)


    # ============================================================
    # Models
    # ============================================================
    """
    Generators use Conv2DTranspose to progressively upsample random noise into realistic images.
    Discriminators uses convolutions to extract features, allow the model to learn complext patterns and textures.
    """
    class ConditionalGenerator(nn.Module):
        def __init__(self, num_classes, z_dim, label_emb_dim):
            super().__init__( )
            self.label_emb = nn.Embedding(num_classes, label_emb_dim)  # build the embedding for the labels.
            # The shape of the label should be (NUM_CLASSES,LABEL_EMBEDDING_DIM})
            input_channels = z_dim + label_emb_dim #The generator input is the label + noise (as a seed)
            """
            The output=(input-1) * stride-2\cdot padding+kernel\_ size
             (1-1) *1-0+7=7
            """
            self.gen = nn.Sequential(
                nn.ConvTranspose2d(input_channels, 256, 7, 1, 0),  # 7x7
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14x14
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 1, 4, 2, 1),  # 28x28
                nn.Tanh() #the gnerator is aligned with data [-1,1], refer to transform above
            )

        def forward(self, noise, labels):
            #The shape of the label embedding should be (NUM_CLASSES,LABEL_EMBEDDING_DIM})
            #Now we chaneg it to 4D shape (batch_size, lable_embedding_dimensions, 1, 1)
            c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
            #The shape of the noise is (curr_batch_size, noise_dimensions, 1, 1)
            x = torch.cat([noise, c], dim=1)
            #concatenate two 4D arrays on the 2nd dimension
            return self.gen(x)

    """
    Generators use Conv2DTranspose to progressively upsample random noise into realistic images.
    Discriminators uses convolutions to extract features, allow the model to learn complext patterns and textures.
    """
    class ConditionalDiscriminator(nn.Module):
        def __init__(self, num_classes, image_size):
            super().__init__()
            self.NUM_CLASSES = num_classes
            self.IMG_SIZE = image_size
            self.label_emb = nn.Embedding(self.NUM_CLASSES, self.IMG_SIZE * self.IMG_SIZE)
            self.disc = nn.Sequential(
                nn.Conv2d(2, 128, 4, 2, 1),  # 28 x 28
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),  # Prevents overfitting on small data
                nn.Conv2d(128, 256, 4, 2, 1), # 14 x 14
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Conv2d(256, 1, 7, 1, 0), # 7 x 7 => 1
                nn.Sigmoid()
            )

        def forward(self, img, labels):
            c = self.label_emb(labels).view(-1, 1, 28, 28)
            #this one will change each label into (1, 1, 28,28) tensor
            #now both img and c have shape (batch, 1, 28,28), combined into (batch, 2, 28, 28)
            x = torch.cat([img, c], dim=1)

            return self.disc(x)


    # ============================================================
    # Training Logic
    # ============================================================
    def train(self):
        gen = self.ConditionalGenerator(self.NUM_CLASSES, self.Z_DIM, self.LABEL_EMB_DIM).to(self.DEVICE)
        disc = self.ConditionalDiscriminator(self.NUM_CLASSES, self.IMG_SIZE).to(self.DEVICE)

        # Using specific betas for GAN stability
        opt_gen = optim.Adam(gen.parameters(), lr=self.G_LR, betas=(0.0, 0.9))
        opt_disc = optim.Adam(disc.parameters(), lr=self.D_LR, betas=(0.0, 0.9))
        criterion = nn.BCELoss()

        print(f"Training on {self.DATA_SIZE} samples...")

        for epoch in range(1, self.EPOCHS + 1):
            for batch_idx, (real, labels) in enumerate(self.loader):
                real, labels = real.to(self.DEVICE), labels.to(self.DEVICE)
                curr_batch_size = real.shape[0]

                # --- Train Discriminator ---
                # many people forgot to put noise to GPU
                noise = torch.randn(curr_batch_size, self.Z_DIM, 1, 1).to(self.DEVICE)
                fake = gen(noise, labels)

                # Label Smoothing (0.9 instead of 1.0)
                # Using 1 it will be over confident
                d_real = disc(real, labels).reshape(-1)
                loss_d_real = criterion(d_real, torch.ones_like(d_real) * 0.9)

                d_fake = disc(fake.detach(), labels).reshape(-1) #flattern
                loss_d_fake = criterion(d_fake, torch.zeros_like(d_fake))

                loss_d = (loss_d_real + loss_d_fake) / 2
                disc.zero_grad()
                loss_d.backward()
                opt_disc.step()

                # --- Train Generator ---
                # We train G to fool D
                output = disc(fake, labels).reshape(-1)
                loss_g = criterion(output, torch.ones_like(output))

                gen.zero_grad()
                loss_g.backward()
                opt_gen.step()


            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{self.EPOCHS}] | D Loss: {loss_d:.4f} | G Loss: {loss_g:.4f}")
                self.save_samples(gen, epoch)

                # --- SAVE MODEL ---
                torch.save(gen.state_dict(), f"{self.MODEL_DIR}/gen_epoch_{epoch}.pth")
                torch.save(disc.state_dict(), f"{self.MODEL_DIR}/disc_epoch_{epoch}.pth")

        return gen


    def save_samples(self, gen, epoch):
        gen.eval()
        with torch.no_grad():
            labels = torch.arange(10).repeat(8).to(self.DEVICE)
            noise = torch.randn(80, self.Z_DIM, 1, 1).to(self.DEVICE)
            samples = gen(noise, labels)
            vutils.save_image(samples, f"{self.OUT_DIR}/epoch_{epoch}.png", nrow=10, normalize=True)
        gen.train()

    def inference(self):
        loaded_gen = self.ConditionalGenerator(self.NUM_CLASSES,self.Z_DIM, self.LABEL_EMB_DIM).to(self.DEVICE)
        # Path to the specific epoch weight you want to load
        weights_path = f"{self.MODEL_DIR}/gen_epoch_50.pth"
        if os.path.exists(weights_path):
            loaded_gen.load_state_dict(torch.load(weights_path, map_location=project.DEVICE))
            loaded_gen.eval()
            # Generate 64 instances of the digit '7'
            for i in range (0,10):
                test_label = i
                with torch.no_grad():
                    noise = torch.randn(64, project.Z_DIM, 1, 1).to(project.DEVICE)
                    labels = torch.full((64,), test_label, dtype=torch.long).to(project.DEVICE)
                    samples = loaded_gen(noise, labels)
                    vutils.save_image(samples, f"../output/cgan_gen_{test_label}.png", normalize=True)
                    print(f"Successfully generated {test_label}s from loaded model!")
        else:
            print("No weights found. Please run training first.")


if __name__ == "__main__":
    project = CGan_DigitGen()
    project.data_preparation()
    project.train()
    print("Training Complete.")
    project.inference()






