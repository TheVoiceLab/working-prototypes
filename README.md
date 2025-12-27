# Projects Descriptions

The following list of active projects highlights the research and development currently underway at the Comcast AI Lab.

All project datasets and model architectures have been specifically optimized for the NVIDIA RTX 5090 (Blackwell) to maximize throughput and efficiency. While CPU execution is supported for testing and inference, please note that training on a CPU is not recommended for production use due to significant performance constraints. For environments lacking a discrete GPU, please utilize the requirements.cpu.txt configuration file for setup."

### Project 1 Comparative Study of cGAN, WGAN‑GP, and Diffusion Models

## [Unsupervised Generative Models] Comparative Study of cGAN, WGAN‑GP, and Diffusion Models
This project implements and compares three major generative modeling approaches—**Conditional GAN (cGAN)**, **Wasserstein GAN with Gradient Penalty (WGAN‑GP)**, and **Diffusion Models**—to understand how different architectures learn data distributions, handle conditioning, and produce synthetic images.

All models are trained on the same dataset under consistent settings to highlight differences in stability, controllability, and sample quality.

<table border="0">
  <tr>
    <td align="center">
      <img src="./unsupervised-learning/diffusion_gan_comparison/data/img.png" width="250"><br>
      <sub><b>Conditional GAN</b></sub>
    </td><td width = 5 />
    <td align="center">
      <img src="./unsupervised-learning/diffusion_gan_comparison/data/pro_epoch_30.png" width="250"><br>
      <sub><b>WGAN‑GP</b></sub>
    </td><td width = 5 />
    <td align="center">
      <img src="./unsupervised-learning/diffusion_gan_comparison/data/img_1.png" width="250"><br>
      <sub><b>Diffusion Model</b></sub>
    </td>
  </tr>
</table>

---

## [Supervised Deep Learning] Guidance Queries for Audio Cache