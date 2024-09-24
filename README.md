# Glass Bottle Segregator

This project implements a Glass Bottle Segregator using a Convolutional Neural Network (CNN) for image classification and an Arduino system for physical sorting of bottles. The system classifies bottles into brown, green, or clear categories based on their color and sorts them into corresponding bins for recycling.

## Project Overview
- **Hardware**: Arduino Uno, servo motors, LEDs.
- **Software**: Python, OpenCV, TensorFlow for neural network.
- **Function**: A camera captures images of the bottles, processes them using a neural network, and activates the servo motors to sort the bottles into the correct bin.

## Project Files
- `Arduino Code/arduino_code.ino`: Code for controlling the sorting system based on the neural network's output.
- `Python Code/image_processing.py`: Python script to process images and prepare them for classification.
- `Python Code/neural_network.py`: Neural network model to classify glass bottles based on their color.
- `Documentation/Report on Glass Bottle Segregator.pdf`: Detailed report of the project, including methodology and results.

## Running the Code
1. Load the `arduino_code.ino` into your Arduino.
2. Use `image_processing.py` to capture images from a camera.
3. Use `neural_network.py` to classify the captured images.
