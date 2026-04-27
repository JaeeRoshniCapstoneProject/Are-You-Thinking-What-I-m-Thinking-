# Are You Thinking What I'm Thinking?: Examining Conceptual Separation in Neural Architectures

This repository contains the code for thesis research investigating how different neural network architectures -CNNs and LLMs -encode semantic information geometrically in their embedding spaces, and how prompting affects that structure.

---

## Research Overview

**Core Research Questions:** 
1. **What Metrics Holistically Assess Models?** We explore several metrics of model performance that interrogate the model's internal representation space and are architecture-aware. 
2. **For Which Concepts Can Models Think Like We Think?** We hypothesised that while there are some concepts that models understand the way humans do, there are many that they struggle to differentiate. Our thesis tries to find a way to identify those cases. 

Two parallel tracks of experiments are conducted:

1. **CNN Experiments** -embedding analysis of image categories using pre-trained vision models
2. **LLM Experiments** -embedding analysis of sentence categories using BERT and GPT, with and without classification prompts

---

## Repository Structure

```
thesis_code/
├── cnns/                          # CNN embedding experiments (image data)
│   ├── Experiment_1_Cats_Dogs_Cars.ipynb
│   ├── Experiment_2_Rangoli_Microscopy.ipynb
│   └── Experiment_3_Pose_Roads.ipynb
│
├── llms/                          # LLM embedding experiments (text data)
│   ├── bert_large/
│   │   ├── bert_w_prompt.ipynb
│   │   ├── bert_wo_prompt.ipynb
│   │   ├── bert_compare_prompted.ipynb
│   │   └── bert_second_moment_analysis.ipynb
│   └── gpt_oss_20b/
│       ├── sentences_with_prompt.ipynb
│       ├── sentences_without_prompt.ipynb
│       ├── compare_prompted.ipynb
│       └── second_moment_analysis.ipynb
│
└── data/
    └── llm_data/
        ├── Sentences.xlsx              # Shakespeare (100) + Computer Science (100)
        ├── comp_sci_sentences.xlsx     # Information Security (100) + Theory of Computation (100)
        └── Hate vs Not-Hate.xlsx       # Hate Speech (100) + No-Hate Speech (100)
```

---

## Experiments

### CNN Experiments (`cnns/`)

Pre-final-layer activations are extracted from **ResNet-50** (2048-d) and **MobileNetV2** (1280-d) on 100 images per class, then analyzed geometrically.

| Notebook | Image Categories |
|---|---|
| Experiment 1 | Cats, Dogs, Cars |
| Experiment 2 | Rangoli, Microscopy (+ Cats, Dogs, Cars for comparison) |
| Experiment 3 | Pose (sleeping/standing), Roads |

**Metrics computed:** PCA 3D visualization, intra/inter-class Euclidean + Cosine + Mahalanobis distances, KL divergence between activation distributions, mean activation histograms.

---

### LLM Experiments (`llms/`)

Embeddings are extracted from **BERT-large-uncased** (1024-d) and **GPT-OSS-20B** (2880-d) for six sentence categories across two conditions:

- **With prompt:** sentence encoded with a classification prompt (e.g., *"Classify this text as Shakespeare or Computer Science:"*); only sentence tokens are pooled
- **Without prompt:** sentence encoded directly

**Sentence categories:**
- Shakespeare
- Computer Science
- Information Security
- Theory of Computation
- Hate Speech
- No-Hate Speech

**Pairwise comparisons analyzed:**
- Shakespeare vs. Computer Science
- Information Security vs. Theory of Computation
- Hate Speech vs. No-Hate Speech

**Metrics computed:** PCA projections (2D/3D), intra/inter-class distance matrices (Euclidean, Cosine, Mahalanobis), KL divergence, silhouette scores, overlap analysis (nearest cross-category sentence pairs).

---

## Setup and Usage

All notebooks are designed to run on **Google Colab** with GPU support (A100 recommended for GPT-OSS-20B experiments).

### Dependencies

```
torch
torchvision
transformers
scikit-learn
scipy
matplotlib
seaborn
pandas
numpy
openpyxl
```

Install via:
```bash
pip install torch torchvision transformers scikit-learn scipy matplotlib seaborn pandas numpy openpyxl
```

### Running Experiments

1. Upload the `data/` directory to your Colab environment or mount Google Drive
2. Open the desired notebook in Colab
3. Update any file paths at the top of the notebook to match your data location
4. Run all cells sequentially

---

## Models Used

| Model | Source | Embedding Dim | Used In |
|---|---|---|---|
| ResNet-50 | torchvision (pretrained) | 2048 | CNN experiments |
| MobileNetV2 | torchvision (pretrained) | 1280 | CNN experiments |
| BERT-large-uncased | Hugging Face | 1024 | LLM experiments |
| GPT-OSS-20B | Hugging Face | 2880 | LLM experiments |

---

## Surpervisor and Authors
Supervisor - Professor Subhashis Banerjee

Authors - Jaee Ponde & Roshni Agarwal
