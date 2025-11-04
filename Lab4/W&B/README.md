# CIFAR-10 CNN Trainer with W&B Integration

A TensorFlow/Keras implementation for training a CNN on the CIFAR-10 dataset with Weights & Biases tracking.

## Overview

This project trains a convolutional neural network to classify CIFAR-10 images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It uses Weights & Biases for experiment tracking, including metrics, visualizations, and model checkpointing.

## Features

- CNN with two convolutional layers and dropout regularization
- Full W&B integration for tracking experiments
- Custom callbacks for logging predictions and confusion matrices
- Automatic learning rate reduction and early stopping
- Model versioning through W&B artifacts

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Weights & Biases
- NumPy

## Installation

Install the required packages:
```bash
pip install tensorflow wandb numpy
```

Set up your W&B account:
```bash
wandb login
```

You can get your API key from https://wandb.ai/authorize

Don't forget to update the API key in the script:
```python
wandb.login(key="your_api_key_here")
```

## Model Architecture

The network structure is pretty straightforward:

- Input: 32x32x3 RGB images
- Conv2D: 32 filters, 3x3 kernel, ReLU
- Conv2D: 64 filters, 3x3 kernel, ReLU
- MaxPooling: 2x2
- Dropout: 30%
- Dense: 128 units, ReLU
- Output: 10 units, Softmax

Uses Adam optimizer with categorical crossentropy loss.

## Configuration

Default hyperparameters:
```python
dropout = 0.3
conv1_filters = 32
conv2_filters = 64
dense_size = 128
learn_rate = 0.001
epochs = 5
batch_size = 64
```

You can change these in the `CIFAR10Trainer.__init__` method by modifying the `cfg` dictionary.

## Custom Callbacks

The code includes three custom callbacks:

**LogLRCallback** - Logs the learning rate each epoch

**LogSamplesCallback** - Creates a table showing 16 test images with their predictions, true labels, and confidence scores

**ConfusionMatrixCallback** - Generates a confusion matrix for the validation set after each epoch

Plus the standard Keras callbacks for reducing learning rate on plateau and early stopping.

## Output Files

The script creates two folders:

- `checkpoints/` - Saves the model after each epoch
- `artifacts/` - Contains the final trained model

All models are also uploaded to W&B for easy access later.

## W&B Dashboard

Once training starts, you can view real-time metrics on your W&B dashboard:

- Training and validation loss/accuracy
- Learning rate over time
- Sample predictions with images
- Confusion matrices
- Model checkpoints

The final model gets saved as a W&B artifact for version control.