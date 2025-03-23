import numpy as np
import pandas as pd
from train import gradient_descent
from utils import get_accuracy, get_predictions, normalize_data
from model import forward_prop

data = pd.read_csv("data/mnist_train.csv")
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = normalize_data(data_dev[1:n])

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = normalize_data(data_train[1:n])

alpha = 0.10
iterations = 500


W1, b1, W2, b2, accuracy_list = gradient_descent(X_train, Y_train, alpha, iterations)

_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
predictions_dev = get_predictions(A2_dev)
accuracy_dev = get_accuracy(predictions_dev, Y_dev)
print(f"Dev Set Accuracy: {accuracy_dev * 100:.2f}%")

import matplotlib.pyplot as plt

plt.plot(accuracy_list)
plt.title('Model Accuracy Over Iterations')
plt.xlabel('Iteration (x10)')
plt.ylabel('Accuracy')
plt.show()
