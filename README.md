# Projects Descriptions

This suite of projects extends our previous research and innovations. The model architectures and datasets were specifically engineered for development on the NVIDIA RTX 5090 (Blackwell) to maximize performance. While the code supports training and inference on a CPU, please be aware that training may take several days due to hardware limitations. If a discrete GPU is unavailable, use the requirements.cpu.txt file for your environment setup.
## Project 1: Comparative Study of Generative Image Models

**Environment:** PyTorch  

**Models Trained:** Conditional GAN (cGAN), Wasserstein GAN (WGAN-GP), Diffusion Models

**Focuse Areas:** Model Stability, Interaction Between Label and Noise Dimenstions

**Project Directory:** [`unsupervised-learning/diffusion_gan_comparison`](./unsupervised-learning/diffusion_gan_comparison)


## Project 2: User Query Prediction and EOS Detection (model part)

**Environment:** PyTorch 

**Models Trained:** Distance and Vocabulary Embeddings, PyTorch Transformer

**Focuse Areas:** Probability Loop, Causal Mask

**Project Directory:** [`generative/sentence_prediction`](./generative/sentence_prediction)


## Project 3: Chronological Audio Classification (Audio Cache)

**Environment:** PyTorch, 

**Models Trained:** CNN, ResNet (CNN), Wav2Vect (Transformer)

**Focuse Areas:** Evaluated performance gains by migrating from traditional convolutional architectures to state-of-the-art transformer-based self-supervised models.

**Project Directory:** [`supervised`](supervised)
