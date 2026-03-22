# Brain Tumour MRI Classifier

A deep learning model that classifies brain MRI scans into 4 categories: **glioma**, **meningioma**, **pituitary**, and **no tumour** — achieving a testing accuracy of **94%**.

## Overview

Brain tumours often go undetected due to their slow growth. This model aids medical practitioners by providing a fast, consistent secondary opinion on MRI scans.

## Model Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Layers:** 3 convolutional layers with increasing channels (3→5→10→15)
- **Kernel size:** 3×3
- **Regularization:** Dropout (p=0.4)
- **Downsampling:** Max pooling
- **Loss Function:** Cross Entropy Loss
- **Best Hyperparameters:** Batch size: 15 | Epochs: 150 | Learning rate: 1e-6

## Dataset

- ~28,000 MRI scans from 3 Roboflow datasets
- Images resized to 224×224 pixels (RGB)
- Balanced across all 4 classes
- Split: 70% train / 20% validation / 10% test

## Results

| Dataset | Accuracy |
|---------|----------|
| Training | ~100% |
| Validation | 94.18% |
| Testing | 94.33% |
| Kaggle (4 images) | 100% |
| Kaggle (20 images) | 95% |

## Limitations

- Lower accuracy on lateral and zoomed-in MRI scans due to imbalanced augmentation
- Transfer learning (VGG16, ResNet, InceptionV3, etc.) proved ineffective for medical imaging

## Ethical Notice

This model is intended as a **secondary diagnostic tool only**. Final diagnosis remains the responsibility of a qualified medical practitioner. Patient data must be handled securely and with consent.
