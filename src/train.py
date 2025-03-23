import numpy as np
from model import init_params, forward_prop, backward_prop, update_params
from utils import get_predictions, get_accuracy, one_hot

def gradient_descent(X, Y, alpha, iterations):
    m = X.shape[1]
    W1, b1, W2, b2 = init_params()
    accuracy_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            accuracy_list.append(accuracy)
            print(f"Iteration: {i}, Accuracy: {accuracy * 100:.2f}%")

    return W1, b1, W2, b2, accuracy_list
