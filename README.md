# Air Hands Number Drawing

A Computer Vision application that lets you draw digits in the air using a colored marker/object via webcam and predicts the drawn digit (0 - 9) in real time using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- CNN Classifier: Built with TensorFlow/Keras to predict handwritten digits (28x28 input shape).
- Color Tracking Air-Canvas: Track any blue target object in real-time using OpenCV HSV color thresholding.
- Interactive Controls: Keybindings to clear the screen, trigger digit prediction, or exit.

## Installation

Ensure you have Python 3.8+ installed. Install the necessary packages via pip:

pip install opencv-python numpy tensorflow matplotlib

## Quick Start Guide

### 1. Train the Model
Run the training script to fetch the MNIST dataset, train the CNN, and save digit_identify.keras:

python train_mnist.py

### 2. Launch the Air Canvas App
Ensure your webcam is connected, then start the main application:

python cv2_thing.py
