# Digit-Classifier
🧠 MNIST Neural Network from Scratch
📚 Overview
This project implements a fully connected neural network from scratch using NumPy to classify handwritten digits from the famous MNIST dataset. The model is built without using high-level libraries such as TensorFlow or PyTorch for gaining a deeper understanding of the fundamental concepts of deep learning, including:

Forward propagation

Backpropagation

Gradient descent

Softmax classification

🚀 Project Highlights


✅ Neural Network Architecture:

Input Layer: 784 neurons (28x28 flattened image)

Hidden Layer: 10 neurons with ReLU activation

Output Layer: 10 neurons with softmax activation

✅ Core Concepts Implemented:

One-hot encoding for labels

Derivation of gradients using backpropagation

Parameter updates using gradient descent

Accuracy evaluation on training and test datasets

✅ Performance:

Achieved ~95% accuracy on the MNIST dataset after 500 iterations.

Visualization of accuracy progression over iterations.

📝 Dataset Information
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), where:

60,000 images are used for training.

10,000 images are used for testing.

Each image is of size 28x28 pixels and is flattened to a vector of size 784.

📊 Model Architecture
Input Layer: 784 features (flattened 28x28 image)

Hidden Layer: 10 neurons with ReLU activation

Output Layer: 10 neurons with softmax activation for multi-class classification

⚡️ How the Model Works
Data Preprocessing:

Normalization of pixel values between 0 and 1.

Splitting data into training and dev sets.

Forward Propagation:

Compute activations through the network.

Use ReLU for the hidden layer and softmax for the output layer.

Loss Calculation:

Use cross-entropy loss for classification tasks.

Backpropagation:

Compute gradients and update parameters using gradient descent.

Model Evaluation:

Compute accuracy on training and test sets.


