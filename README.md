# DeepTyphoon-Enhanced

## Overview & Acknowledgement

This project is an enhanced and refactored version of the original project  
**deep_typhoon** by melissa135:  
https://github.com/melissa135/deep_typhoon  

The original repository provides a baseline CNN-based framework for typhoon intensity estimation from satellite imagery.

This enhanced version focuses on:

- Improving data splitting methodology  
- Optimizing pretrained weight transfer efficiency  
- Enhancing model generalization and training stability  

All improvements were implemented independently while preserving the original research objective.


## Key Improvements 

### Weight Cloning Strategy

To adapt **ResNet-18** for 2-channel satellite imagery:

- Modified the first convolutional layer to accept 2 input channels  
- Implemented partial weight cloning from ImageNet pretrained weights  
- Preserved low-level visual feature representations  



### Rigorous Data Splitting

Implemented a strict three-way split



### Performance Optimization

To improve training stability and generalization:

- Added Dropout layers  
- Introduced StepLR learning rate scheduler  
- Tuned hyperparameters for convergence stability  


## Performance Metrics

| Metric | Value |
|--------|--------|
| R² Score | 0.8754 |
| MAE | 6.88 m/s |
| RMSE | 8.99 m/s |


## Requirements

Python 3.11
matplotlib==3.10.8
numpy==2.4.2
opencv_python==4.12.0.88
Pillow==12.1.1
scikit_learn==1.8.0
torch==2.6.0+cu124
torchvision==0.21.0+cu124
