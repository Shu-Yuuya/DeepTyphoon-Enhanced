# DeepTyphoon-Enhanced

## Overview

This project is an enhanced and refactored version of the original project  
**deep_typhoon** by melissa135:  https://github.com/melissa135/deep_typhoon  
The original repository provides a baseline CNN-based framework for typhoon intensity estimation from satellite imagery.
The dataset is sourced from agora/JMA

## Core Refactoring & Optimizations

This project underwent a comprehensive architectural overhaul and workflow optimization to mitigate overfitting and enhance the predictive accuracy for typhoon intensity. The key improvements are detailed below:

### Model Architecture Evolution
* Upgraded the baseline custom CNN to a **ResNet-18** architecture.
* Incorporated **ImageNet pre-trained weights** for initialization. 
* Tailored the ResNet input layer to accommodate the specific 2-channel nature of the typhoon data. 

### Training Strategy & Generalization
* Integrated **Dropout** layers within the fully connected head and applied **L2 Regularization** (Weight Decay) to the optimizer. 
* Implemented a **StepLR** scheduler that automatically decays the learning rate every 4 epochs. 
* Introduced an automated early stopping mechanism that monitors validation metrics in real-time. 

### Data Engineering & Evaluation Framework
* Transitioned from a single Loss-based evaluation to a comprehensive suite including **MAE**, **RMSE**, and **$R^2$ Score**. The **$R^2$ Score** is now utilized as the primary criterion for selecting the "Best Model".
* Utilized `random_split` to strictly divide the data into **Training, Validation, and Test** sets. 
* Separated the preprocessing into `train_transform` and `test_val_transform`. The training pipeline specifically includes **restricted 30° rotations** and random flips.

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
## Visualizations
<img width="1000" height="500" alt="Image" src="https://github.com/user-attachments/assets/dfda2687-8963-4746-b2e9-35ba08965cda" />
<img width="1000" height="500" alt="Image" src="https://github.com/user-attachments/assets/1929a5d3-9abb-418a-b883-2af64d900c28" />
<img width="800" height="800" alt="Image" src="https://github.com/user-attachments/assets/1c0f02d9-dda0-47da-a128-fc37807cc7ab" />
