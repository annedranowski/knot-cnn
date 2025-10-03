# Data

Source of truth: Hugging Face dataset `tr33hugg3r/knot-crossings`.

## Sync
```bash
python -m pip install -r requirements.txt
python scripts/sync_hf_data.py --repo-id tr33hugg3r/knot-crossings --repo-type dataset --revision main

---

## 📂 Dataset Structure

The dataset is organized into two splits:

```
train/
0/
3/
4/
...
test/
0/
3/
4/
...
```

- Each folder corresponds to a **crossing number** class.
- For example, `3/` contains diagrams of **prime knots with 3 crossings**.

---

## ✅ Classes

Each class label is the **minimum crossing number** of the knot:

| Label | Meaning |
|-------|---------|
| `0`   | Unknot |
| `3`   | Trefoil knots |
| `4`   | Figure-eight knots |
| ...   | Higher crossing numbers |

---

## 🔢 Dataset Statistics

| Split | # Images | Classes |
|-------|--------:|---------|
| Train | **7,240** | 0, 3, 4, 5, 6, 7, 8, 9, 10 |
| Test  | **1811**   | 0, 3, 4, 5, 6, 7, 8, 9, 10, 11 |

- The **train split** covers 9 different crossing-number classes.
- The **test split** includes a few additional samples of higher crossings (10, 11).

---

## 🛠️ How to Load

Once published, you can load it directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("annedranowski/knot-crossings")
dataset["train"][0]


## 🎯 Intended Use
This dataset is intended for:

Training and evaluating CNN or Vision Transformer (ViT) models to classify knot diagrams.

Studying computer vision approaches for topological diagram recognition.

Exploring ML methods for mathematical objects and low-dimensional topology.

It is not intended for real-world industrial vision tasks—rather it is a research-oriented dataset.

---

📜 License
MIT License

---

✨ Citation

@dataset{dranowski2025knot,
  title     = {Knot Crossing Dataset},
  author    = {Anne Dranowski and Yura Kabkov and Roman Melamud and Daniel Tubbenhauer},
  year      = {2025},
  url       = {https://huggingface.co/datasets/annedranowski/knot-crossings}
}

---

📂 TRAIN SPLIT
  Class   0: 1029 images
  Class  10: 1008 images
  Class   3: 1029 images
  Class   4: 1029 images
  Class   5: 1008 images
  Class   6: 1008 images
  Class   7: 1029 images
  Class   8: 882 images
  Class   9: 1029 images
  ➡ TOTAL: 9051 images


📂 TEST SPLIT
  Class   0: 3 images
  Class  10: 6 images
  Class  11: 2 images
  Class   3: 4 images
  Class   4: 4 images
  Class   5: 6 images
  Class   6: 6 images
  Class   7: 6 images
  Class   8: 6 images
  Class   9: 6 images
  ➡ TOTAL: 49 images
