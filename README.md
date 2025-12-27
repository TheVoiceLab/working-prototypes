# Projects Descriptions

Below is a list of working projects that demonstrates the creation work we have been developing at comcast AI lab.
The workspace had been optimized for demo with blackwell RTX5090 GPU, though a few of them run better with H100.
However, you can also use CPU to test the code outhe model reference part should be ok only the trainning part can be very slow.
requirements.cpu.txt has CPU version configuration.

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