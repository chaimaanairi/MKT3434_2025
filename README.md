# MKT3434 - Assignment #3

## Model Builder: Interactive Neural Network Designer

### Dr. Ertuğrul BAYRAKTAR  
### Chaimaa Nairi - 23501057  

---

## Overview  
An intuitive GUI-based tool to build, train, and evaluate deep learning models with support for Dense, CNN, and RNN layers, plus training enhancements and model persistence.

---

## Features

### Model Architecture
- **Configurable Dense Layers:** Add/remove layers, set units (e.g., 128 → 64 → 32), choose activations (ReLU, Sigmoid, Tanh).
- **CNN Support:** Add convolutional layers (custom filters, kernel size) and pooling layers (max/average), ideal for image data (e.g., MNIST).
- **RNN Support:** Add LSTM/GRU layers for sequential data.

### Model Persistence & Management
- Save/load model architecture and weights (`.json`, `.h5`).
- Dynamic layer management with GUI buttons for adding/removing layers.

### Training Enhancements
- Optimizers: Adam, SGD, RMSprop (selectable).
- Learning rate schedulers: Step decay, exponential decay.
- Regularization: Dropout, L2.
- Early stopping based on validation loss.
- Real-time training visuals: Loss/accuracy plots and gradient histograms.

---