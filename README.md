# Lab 2: Deep Learning with PyTorch for Computer Vision

## Overview
This lab explores the implementation of various deep learning architectures for image classification on the MNIST dataset. Using PyTorch, the lab compares Convolutional Neural Networks (CNN), pre-trained models (VGG16 and AlexNet), and Vision Transformers (ViT). The performance of these models is analyzed through metrics such as accuracy, F1-score, training time, and loss.

The Faster R-CNN architecture was also explored but noted for its computational intensity, requiring significant resources.

## Objectives
The primary goals of this lab include:
- Designing and training CNN-based architectures.
- Fine-tuning pre-trained models (VGG16, AlexNet) for the MNIST dataset.
- Exploring Vision Transformers and their implementation for classification tasks.
- Comparing model performance through detailed metrics and interpreting the results.

## Dataset
**MNIST Dataset**  
- URL: [Kaggle - MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
- A collection of 70,000 grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels.

## Implemented Models and Results

### 1. Convolutional Neural Network (CNN)
- **Loss during training**:
- Epoch 1: Loss = 0.0058 Epoch 2: Loss = 0.0039 Epoch 3: Loss = 0.0073 Epoch 4: Loss = 0.0052 Epoch 5: Loss = 0.0048
- **Test Accuracy**: 99.12%  
- **Classification Report**:  
| Precision | Recall | F1-Score | Support |
|-----------|--------|----------|---------|
| 0.99      | 1.00   | 0.99     | 980     |
| 0.99      | 1.00   | 1.00     | 1135    |
| 0.99      | 0.99   | 0.99     | 1032    |
| ...       | ...    | ...      | ...     |
| **Overall Accuracy** | **0.99** | **Weighted Avg: 0.99** |

---

### 2. VGG16 (Fine-tuned Pre-trained Model)
- **Loss during training**:
- Epoch 1: Loss = 1.1196 Epoch 2: Loss = 0.0753 Epoch 3: Loss = 0.0425 Epoch 4: Loss = 0.0337 Epoch 5: Loss = 0.0257
- **Training Time**: 2273.53 seconds  
- **Accuracy**: 99.11%  
- **F1-Score**: 99.11%

---

### 3. AlexNet (Fine-tuned Pre-trained Model)
- **Loss during training**:
- Epoch 1: Loss = 0.5870 Epoch 2: Loss = 0.0667 Epoch 3: Loss = 0.0460 Epoch 4: Loss = 0.0351 Epoch 5: Loss = 0.0287
- **Training Time**: 264.15 seconds  
- **Accuracy**: 99.31%  
- **F1-Score**: 99.31%

---

### 4. Faster R-CNN
- Noted for its high computational resource requirements and extended training time. Detailed results could not be obtained due to resource constraints.

---

### 5. Vision Transformer (ViT)
- **Loss during training**:
- Epoch 1: Loss = 1.2380 Epoch 2: Loss = 0.3352 ... Epoch 19: Loss = 0.0352 Epoch 20: Loss = 0.0329
- **Test Accuracy**: 99.23%  
- **Observations**: ViT demonstrated competitive performance compared to CNNs and pre-trained models while providing a novel transformer-based architecture.

---

## Comparative Analysis
| Model           | Accuracy | F1-Score | Training Time (seconds) | Notes                                         |
|------------------|----------|----------|--------------------------|-----------------------------------------------|
| CNN             | 99.12%   | 99.12%   | -                        | Custom architecture, fast and efficient.      |
| VGG16           | 99.11%   | 99.11%   | 2273.53                  | Pre-trained, slower due to complexity.        |
| AlexNet         | 99.31%   | 99.31%   | 264.15                   | Fast training, pre-trained.                   |
| Vision Transformer | 99.23%   | 99.23%   | -                        | Excellent performance with novel architecture.|
| Faster R-CNN    | -        | -        | -                        | Computationally intensive, limited by resources.|

## Tools & Resources
- **Framework**: PyTorch  
- **Environment**: Google Colab, Kaggle  
- **References**:  
- [ViT Paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)  
- [Vision Transformers Tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)  

## Insights and Conclusion
This lab highlights the trade-offs between custom architectures (CNN), pre-trained models (VGG16, AlexNet), and modern transformer-based models (ViT). While CNNs provide simplicity and efficiency, pre-trained models offer ease of fine-tuning, and ViT introduces state-of-the-art methodologies for vision tasks.

Future improvements could involve:
- Utilizing larger datasets for fine-tuning.
- Exploring distributed training for Faster R-CNN and ViT.
- Applying advanced optimization techniques for better efficiency.
