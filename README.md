## Plant Disease Classification using CNN (TensorFlow & Keras)

A deep learning–based image classification project that detects healthy and diseased tomato plant leaves using Convolutional Neural Networks (CNN). This project uses a subset of the PlantVillage dataset and demonstrates how artificial intelligence can be applied to smart agriculture and crop disease monitoring.

## Project Overview

Plant diseases greatly reduce agricultural productivity. Early and accurate disease detection allows farmers to take timely action and prevent crop loss.

In this project, a CNN model is trained to automatically identify whether a tomato leaf is healthy or affected by a specific disease. The model is evaluated using accuracy and loss graphs along with analysis of misclassified samples.

# Dataset

The dataset used in this project is taken from the PlantVillage dataset available on Kaggle.

Approximately ten tomato leaf classes were selected from more than 54,000 images. The data was divided into 80% for training and 20% for validation.

Some of the classes include:

Tomato Healthy

Tomato Early Blight

Tomato Late Blight

Tomato Leaf Mold

Tomato Septoria Leaf Spot

Tomato Yellow Leaf Curl Virus

Tomato Bacterial Spot

# Model Architecture

A Convolutional Neural Network was built using TensorFlow and Keras. The model consists of multiple convolutional layers for feature extraction, followed by max pooling layers for dimensionality reduction. ReLU activation is used throughout the network.

The extracted features are flattened and passed through fully connected dense layers. A softmax output layer is used for multi-class classification.

# Training Details

The model was implemented using TensorFlow and Keras. Training was performed for 10 epochs with a batch size of 32. The Adam optimizer was used along with the categorical crossentropy loss function. All images were resized to 128 × 128 pixels before training.

# Results

Training and validation accuracy and loss were plotted to evaluate model performance. The model successfully learned to distinguish between healthy and diseased leaves. Misclassified samples were also analyzed to understand the limitations of the model.

# Real-World Application

This system can be integrated into mobile applications for farmers, smart farming IoT systems, drone-based crop monitoring, and automated disease detection platforms.

# Technologies Used

Python

TensorFlow

Keras

NumPy

Matplotlib

PlantVillage Kaggle Dataset

