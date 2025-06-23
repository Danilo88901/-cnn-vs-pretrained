# 🐑 Sheep Breed Classification with ConvNeXt

This project uses a pretrained ConvNeXt-Base model (via transfer learning) to classify sheep breeds from images as part of the [Sheep Classification Challenge 2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025).

Best validation F1-score achieved: **93%**

---

## 🚀 Model Overview

- **Backbone**: ConvNeXt-Base (pretrained on ImageNet)
- **Classifier Head**: Custom fully connected layers
- **Task**: Multi-class classification of sheep breeds
- **Metric**: Macro F1-score
- **Training Strategy**:
  - First freeze ConvNeXt backbone
  - Then unfreeze after several epochs for fine-tuning
  - Data augmentations + learning rate scheduling


## 🧠 Requirements

Install dependencies via pip:

```bash
pip install torch torchvision scikit-learn pandas matplotlib
