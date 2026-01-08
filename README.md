# med-VQA-assignment
Medical VQA project for Advanced Machine Learning course
# Medical Visual Question Answering (Med-VQA) with Deep Learning

**Course:** WOA7015 - Advanced Machine Learning (Alternative Assessment)  
**Student:** DADDI ADDOUN Sami  
**Matrix No:** 24223007  
**Group:** Rad  

## Project Overview
This project aims to solve the Medical Visual Question Answering (Med-VQA) task, where an AI system must answer clinical questions based on medical images (X-rays, CT scans, MRIs). The goal is to demonstrate the necessity of multimodal fusion (combining vision and language) compared to image-only approaches.

The project is structured in three phases using the **VQA-RAD dataset**:
1.  **CNN Baseline:** An image-only classification model (ResNet50) to establish a lower bound.
2.  **Initial VLM:** A standard fusion model (ResNet50 + LSTM + Word2Vec) trained on the top 15 answers.
3.  **Optimized VLM (SOTA):** An advanced architecture (Partially Fine-tuned ResNet50 + PubMedBERT) trained on an expanded label set (31 classes).

##  Repository Structure

- `CNN_baseline.ipynb`: Implementation of the image-only ResNet50 baseline.
- `VLM_1.ipynb`: First iteration of the VLM using LSTM and frozen Word2Vec embeddings.
- `vlm-opti.ipynb`: **Final Optimized Model** using PubMedBERT and partial fine-tuning of ResNet50.
- `comparison_analysis.ipynb`: Notebook used to compare metrics and generate evaluation plots.
- `results/`: Contains saved training histories and confusion matrices (images).

##  Methodologies

### 1. Data Processing
- **Images:** Resized to 224x224, normalized, and augmented (rotation, crop) for the optimized model.
- **Text:** - *VLM 1:* Tokenization + Pre-trained **Word2Vec** (Google News) embeddings.
    - *Optimized VLM:* BERT Tokenizer + Pre-trained **PubMedBERT** embeddings (Biomedical domain).
- **Strategy:** 5-Fold Stratified Cross-Validation was used for all experiments to ensure robustness.

### 2. Model Architectures
| Model | Image Encoder | Text Encoder | Fusion |
|-------|---------------|--------------|--------|
| **CNN Baseline** | ResNet50 (Frozen) | None | N/A |
| **VLM 1** | ResNet50 (Frozen) | LSTM (Word2Vec) | Element-wise Multiply |
| **Optimized VLM** | ResNet50 (Fine-tuned) | PubMedBERT | Concatenation + Dense Block |

## Key Results
- The **CNN Baseline** struggled to differentiate answers without the question context (~58% Accuracy).
- The **Optimized VLM** significantly outperformed the baseline and the initial VLM by leveraging biomedical language understanding and fine-tuned visual features (~66% Accuracy).
- The use of **PubMedBERT** proved crucial for handling medical terminology compared to generic embeddings.

## Execution Environments

Please note that the notebooks were developed and executed in different environments to optimize resource usage and data access.

* **Google Colab:** Used for `CNN_baseline.ipynb`, `VLM_1.ipynb`, and `comparison_analysis.ipynb`.
* **Kaggle Kernels:** Used for `vlm-opti.ipynb`.
    * *Reason:* Leveraged Kaggle's native access to the VQA-RAD dataset and stable GPU environment for the heavier BERT-based model.

**Important:**please update the `BASE_PATH` variable in the configuration cell to match your local dataset location.

## Requirements
To run the notebooks, the following libraries are required:
```python
tensorflow
transformers  # For BERT
gensim        # For Word2Vec
scikit-learn
opencv-python
nltk
seaborn
