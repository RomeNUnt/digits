# Air Hands Number Drawing

A Computer Vision application that lets you draw digits in the air using a colored marker/object via webcam and predicts the drawn digit (0 - 9) in real time using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- CNN Classifier: Built with TensorFlow/Keras to predict handwritten digits (28x28 input shape).
- Color Tracking Air-Canvas: Track any blue target object in real-time using OpenCV HSV color thresholding.
- Interactive Controls: Keybindings to clear the screen, trigger digit prediction, or exit.
