# 🪢 Knot Recognition with CNN

This repository contains a lightweight convolutional neural network (CNN) for **detecting knots in images**.  
It includes:  
- A **training pipeline** (`src/train.py`)  
- **Dataset utilities** (`src/dataset.py`)  
- A **model definition** (`src/models.py`)  
- A **clean demo notebook** (`notebooks/knot_detector.ipynb`)  

The project is structured for easy **local training**, **evaluation**, and future **deployment on Hugging Face Spaces**.

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/annedranowski/knot-cnn.git
cd knot-cnn
```

## HF integration 

Script to snapshot the HF dataset in data/raw/....

<!-- Point your dataloader to that local path. -->

<!-- When the HF dataset updates, re‑run the sync with a new revision (tag/commit) and you’re reproducible. -->

Run 

```
python scripts/sync_hf_data.py \
  --repo-id tr33hugg3r/knot-crossings \
  --revision main
```

<!-- If your HF dataset ever becomes private, huggingface-cli login once and the script will use your token. -->