# Chest-X-Ray-Pneumonia-Detector

Chest X‑Ray Pneumonia Detector
A lightweight CNN trained on the Kaggle Chest X‑Ray Pneumonia dataset and deployed entirely in the browser using ONNX Runtime Web.

# Dataset
Uses the Kaggle chest X‑ray dataset with a known class imbalance (more PNEUMONIA than NORMAL). Images are resized to 224×224, converted to grayscale, and normalized.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Model
Custom CNN with four convolutional blocks, global average pooling, and a two‑layer classifier (~422k parameters). Designed for fast, client‑side inference.

# Training
Balanced using WeightedRandomSampler and class‑weighted loss.
Optimized with Adam + LR scheduler.
Best validation accuracy: 87.5%.

# Evaluation
Test performance is balanced across classes, with strong recall for NORMAL and strong precision for PNEUMONIA. Confusion matrix and classification report included in the notebook.

# Dataset Limitations
Some images are mislabeled or visually ambiguous.
Scanner quality varies across sources.
Subtle pneumonia cases may resemble normal lungs.
Strong class imbalance requires correction.

# Model Limitations
May miss very subtle pneumonia.
Slight bias toward NORMAL predictions due to dataset noise.
Not a clinical diagnostic tool.

ONNX Deployment
Model exported to ONNX (~1.6 MB) and verified to match PyTorch outputs.
Runs fully in the browser with no backend.

https://jed1soccerkid.github.io/Chest-X-Ray-Pneumonia-Detector/
