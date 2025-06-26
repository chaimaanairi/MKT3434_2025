# MKT3434 - Assignment #3

## Model Builder: Interactive Neural Network Designer

### Dr. Ertuğrul BAYRAKTAR  
### Chaimaa Nairi - 23501057  

---

## Overview  
This project delivers an interactive GUI application built with **PyQt6** that enables users to design, train, and manage neural networks visually. It integrates a deep learning backend with **TensorFlow/Keras** to build models dynamically, train on the MNIST dataset, visualize training progress, and save/load models.

The tool is designed for both beginners and experts, making neural network experimentation accessible without coding.

---

## Features

### Neural Network Architecture  
- **Layer Types:** Dense, Conv2D, LSTM, GRU.  
- **Configurable Parameters:** Units/filters, kernel size (for Conv2D), activation functions (ReLU, Sigmoid, Tanh), dropout rates.  
- **Dynamic Model Building:** Sequential assembly of layers into a Keras model.

### Training & Visualization  
- Train models interactively on MNIST for a fixed number of epochs (default 5).  
- Real-time training logs and final metrics displayed.  
- Visualization of accuracy and loss curves over epochs via Matplotlib.

### Model Management  
- Save models as `.h5` files and load pre-trained models.  
- Layer management via intuitive GUI controls (add/remove).  
- Results and logs shown in a read-only text area.

---

## User Interface

- **Architecture Section:**  
  Dropdown menus and input fields to select layer types and parameters, add layers, and compile the model.

- **Training Section:**  
  Buttons to start training, display training curves, save/load models, and view logs and results.

---
## Conclusion

This Custom Neural Network GUI offers a streamlined, interactive environment to build, train, and manage neural networks. It serves as a practical learning and experimentation tool in deep learning, with scope for future enhancements such as support for additional datasets, more layer types, and asynchronous training with progress indicators.

