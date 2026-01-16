# Projects Descriptions

### This suite of projects extends our previous research and innovations during daily work. 

The model architectures and datasets were specifically engineered for development on the NVIDIA RTX 5090 (Blackwell) to maximize performance. While the code supports training and inference on a CPU, please be aware that training may take several days due to hardware limitations. If a discrete GPU is unavailable, use the requirements.cpu.txt file for your environment setup.
## Project 1: Comparative Study of Generative Image Models

**Environment:** PyTorch

**Business Needs:** When a new voice device is released, we need to retrain our voice models (wake-word spotting and ASR) using utterances that reflect the device’s new acoustic characteristics. However, early on, we typically don’t have sufficient real data. To address this, we use synthetic data generation tools to create additional utterances that simulate the device’s behavior. For privacy reasons, the demo uses images instead of voice spectrograms.

**Models Trained:** Conditional GAN (cGAN), Wasserstein GAN (WGAN-GP), Diffusion Models

**Focuse Areas:** Model Stability, Interaction Between Label and Noise Dimenstions

**Project Directory:** [`unsupervised-learning/diffusion_gan_comparison`](./unsupervised-learning/diffusion_gan_comparison)


## Project 2: User Query Prediction and EOS Detection (model part)

**Environment:** PyTorch 

**Models Trained:** Distance and Vocabulary Embeddings, PyTorch Transformer

**Business Needs:** For hands-free devices, the system often needs to wait for a relatively long period to determine whether the user has finished speaking, which can lead to a sluggish user experience. To mitigate this, we use predictive models to estimate when a user’s command will end and what the command is likely to be, allowing us to begin processing the request ahead of time. Once the actual utterance is confirmed, the VREX result is already prepared and can be returned immediately.

**Focuse Areas:** Probability Loop, Causal Mask

**Project Directory:** [`generative/sentence_prediction`](./generative/sentence_prediction)


## Project 3: Chronological Audio Classification (Audio Cache)

**Environment:** PyTorch, 

**Business Needs:** We observed that popular short queries—such as “YouTube” or “tune to CNN”—account for a large portion of voice remote traffic. Instead of routing these requests through the full ASR pipeline, we directly generate a mel-spectrogram for the short utterance and use a lightweight classifier to determine whether (and which) popular query it matches. This approach delivers near-instant response times and significantly higher accuracy than traditional ASR systems, which often struggle with very short utterances.

For demo purposes, we use publicly available image data to stand in for the actual mel-spectrograms derived from user utterances.

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

**Business Needs:** In our IVR system, we need to map user utterances to predefined intents. However, the system contains thousands of intents, each with detailed descriptions, examples, and multiple versions, making it impractical to rely solely on an LLM for direct intent selection.

To address this, we built a vector-based retrieval system to narrow the search space to a small set of candidate intents, and then use an LLM to make the final determination. A key challenge is that FAISS performs poorly on very short utterances (fewer than three words). To overcome this, we designed two separate vector systems: one that matches short utterances against example phrases for each intent, and another that matches longer utterances against full intent definitions. The LLM then evaluates the combined candidate set to select the best-fitting intent.

**Models Used:** SentenceTransformer (SBERT), MiniBatchKMeans, GPT-5o-mini (Audit)

**Focuse Areas:** Designed a high-throughput pipeline to process 1M+ utterances by condensing raw text into 100 behavioral archetypes, reducing LLM token costs by 99.9%.
Hybrid Analysis Engine: Combined local GPU-accelerated embeddings (PyTorch) for large-scale data structuring with LLM-as-a-Judge logic for nuanced semantic auditing.

**Project Directory:** [`adv_analysis_utterance_classifier`](adv_analysis_utterance_classifier)

