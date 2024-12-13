# Handwritten Text Recognition using EMNIST and Hybrid Techniques

This project implements and compares various machine learning approaches for handwritten character recognition using the EMNIST dataset. We evaluate both deep learning models (CNNs, MLPs) and classical machine learning techniques, analyzing their performance and generalizability.

## Overview
This project focuses on handwritten character recognition using various machine learning approaches. The main objectives are:
- Implementing and comparing different machine learning models
- Evaluating model performance on the EMNIST dataset
- Testing model generalizability using external datasets
- Analyzing the impact of dimensionality reduction techniques

## Models Implemented
- **Deep Learning Models**
  - Multi-Layer Perceptron (MLP)
    - 10 hidden layers
    - Hidden dimensions: 28×28×2 (1568 neurons per layer)
  - Convolutional Neural Network (CNN)
    - Multiple convolutional layers with varying channel sizes
    - Dropout and batch normalization for regularization

- **Classical Machine Learning Models**
  - Logistic Regression
  - Random Forest
  - Quadratic Discriminant Analysis (QDA)


### Prerequisites
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy


## Dependencies
```python
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.2
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.2
```

## License
[MIT License](LICENSE)

## Authors
- Prajwal Moharana
- Pranav Kallem
- Tejas Chandramouli
- Jayden Lim

## Acknowledgments
- EMNIST dataset creators
- PyTorch team
- scikit-learn community
