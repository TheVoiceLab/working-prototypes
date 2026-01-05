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

**Focuse Areas:** Evaluated performance gains by migrating from traditional convolutional architectures to state-of-the-art Wave2Vec transformer-based self-supervised models.

| Model          | Accuracy  | Avg Precision | Avg Recall |
|----------------|-----------|---------------|------------|
| SpeechCNN      | 97.37%    | 97.38%        | 97.37%     |
| ResNet18_Opt   | 97.57%    | 97.59%        | 97.55%     |
| **Transformer**| **98.87%**| **98.87%**    | **98.86%** |

**Project Directory:** [`supervised`](supervised)

## Project 4: Scalable NLU Audit via Semantic Compression

**Environment:** PyTorch, OpenAI, Hybrid Pipeline (Classical ML + Generative AI)

**Models Trained:** SentenceTransformer (SBERT), MiniBatchKMeans, GPT-5o-mini (Audit)

**Focuse Areas:** Designed a high-throughput pipeline to process 1M+ utterances by condensing raw text into 100 behavioral archetypes, reducing LLM token costs by 99.9%.
Hybrid Analysis Engine: Combined local GPU-accelerated embeddings (PyTorch) for large-scale data structuring with LLM-as-a-Judge logic for nuanced semantic auditing.

**Project Directory:** [`semantic_compression_audit`](semantic_compression_audit)

