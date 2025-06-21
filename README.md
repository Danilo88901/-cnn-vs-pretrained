# CNN vs Pretrained ConvNeXt: Image Classification Benchmark

## Overview

This project compares the performance of a custom-built Convolutional Neural Network (CNN) and a pretrained ConvNeXt model on an image classification task using a dataset of sheep breeds.

Both models were trained and evaluated using PyTorch, with macro F1-score as the primary metric. Training and validation results are visualized for easy comparison.

## Key Features

- 🧠 Custom CNN architecture built from scratch
- ⚙️ Transfer learning with pretrained ConvNeXt (ImageNet weights)
- 📊 Visualizations for:
  - Training & validation loss
  - Macro F1-score over epochs
- 🧪 Dataset: Multi-class sheep breed images
- 🔧 Optimized with Adam optimizer and StepLR scheduler

## Folder Structure
This repository contains:
- custom CNN model script
- pretrained ConvNeXt model script
- scripts for training and visualization
- data files and plots
Results
The pretrained ConvNeXt model significantly outperforms the custom CNN, achieving a notably higher macro F1 score, particularly on the validation dataset. This demonstrates the power of transfer learning and pretrained weights.

Training and validation losses, as well as macro F1-score curves for both models, are included to provide a detailed comparison of their learning dynamics.
