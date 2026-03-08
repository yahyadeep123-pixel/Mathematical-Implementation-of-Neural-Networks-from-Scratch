import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- 1. Core Mathematical Functions ---
def ReLU(Z):
    return np.maximum(0, Z)


def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# --- 2. Forward and Backward Propagation ---
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_propagation(Z1, A1, Z2, A2, W2, X, Y):
    M = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / M * dZ2.dot(A1.T)
    db2 = 1 / M * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / M * dZ1.dot(X.T)
    db1 = 1 / M * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


# --- 3. Evaluation and Training ---
def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initialize_parameters()
    print(f"Starting training ({iterations} iterations)...")

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 100 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy:.4f}")

    return W1, b1, W2, b2


# --- 4. Main Execution Function ---
def main():
    # 1. Load Data
    try:
        data = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("\n🛑 ERROR: 'train.csv' file not found.")
        print("Please ensure it is in the same directory as main.py.")
        return

    data = np.array(data)
    np.random.shuffle(data)

    # 2. Split Data and Scale
    M_val = int(data.shape[0] * 0.1)

    data_train = data[M_val:].T
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.0

    data_val = data[:M_val].T
    Y_val = data_val[0]
    X_val = data_val[1:] / 255.0

    # 3. Train
    alpha = 0.10
    iterations = 1000

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations)

    # 4. Evaluate and Display Result
    _, _, _, A2_val = forward_propagation(X_val, W1, b1, W2, b2)
    predictions_val = get_predictions(A2_val)
    accuracy_val = get_accuracy(predictions_val, Y_val)

    print("-" * 30)
    print(f"Final Validation Accuracy: {accuracy_val:.4f}")

    # Display a random example
    test_index = np.random.randint(0, M_val)
    current_image = X_val[:, test_index, None]
    label = Y_val[test_index]

    _, _, _, A2 = forward_propagation(current_image, W1, b1, W2, b2)
    prediction = get_predictions(A2)

    print(f"Predicted for image {test_index}: {prediction[0]}")
    print(f"Actual Label: {label}")
    
    # Display the image
    plt.gray()
    plt.imshow(current_image.reshape((28, 28)) * 255, interpolation='nearest')
    plt.title(f"Predicted: {prediction[0]}, Actual: {label}")
    plt.show()


if __name__ == '__main__':
    main()