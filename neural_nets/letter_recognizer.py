import pandas as pd
import numpy as np
from math import exp
from matplotlib import pyplot as plt
import os
from pathlib import Path

csv_path = Path(__file__).parent.parent / "data" / "A_Z Handwritten Data.csv"
print(csv_path.resolve())
print(csv_path.exists())

data = pd.read_csv(csv_path)
EPOCHS = 1200 # Change depending on how many training runs you wanna push through
data = np.array(data) # array of letters, in number form

letters = {
    0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I',
    9 : 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R',
    18 : 'S', 19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z'
}

def convert_to_letter(num):
    return letters.get(num, '?')  # Returns '?' if num isn't in the dictionary

examples, features = data.shape

# shuffling data so its not in order (only shuffles rows)
np.random.shuffle(data)

# Grabbing first 1k examples for testing only
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:features]
X_dev = X_dev / 255

# Grabbing the rest of the examples for training
data_train = data[1000:60000].T  # Use only 60k instead of 370k examples
Y_train = data_train[0] # basically the answers
X_train = data_train[1:features] # the pixels
X_train = X_train / 255 # getting val between 1 and 0
_, m_train = X_train.shape # The # of training examples (columns)

def init_params():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(26, 64) - 0.5
    b3 = np.random.rand(26, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def deriv_ReLU(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# Y - Labels/answers
# dZ3, dZ2, dZ3 - Gradient error vals
def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    m = X.shape[1]
    dZ3 = A3 - one_hot_Y # error values for final layer
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

# Simply updating params
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Putting it all together
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3

# Loading the previous training data if it exists, if not, train from scratch
if all(os.path.exists(f'./model/{w}.npy') for w in ['W1','b1','W2','b2','W3','b3']):
    print("Loading trained model...")
    W1 = np.load('./model/W1.npy')
    b1 = np.load('./model/b1.npy')
    W2 = np.load('./model/W2.npy')
    b2 = np.load('./model/b2.npy')
    W3 = np.load('./model/W3.npy')
    b3 = np.load('./model/b3.npy')
else:
    print("Training model from scratch...")
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.10, EPOCHS)
    os.makedirs('model', exist_ok=True)
    np.save('./model/W1.npy', W1)
    np.save('./model/b1.npy', b1)
    np.save('./model/W2.npy', W2)
    np.save('./model/b2.npy', b2)
    np.save('./model/W3.npy', W3)
    np.save('./model/b3.npy', b3)
    

# Choosing a random sample from testing data
index = np.random.randint(X_dev.shape[1])
x = X_dev[:, index].reshape(-1, 1)  # input must be column vector
true_label = Y_dev[index] 

# Forward propgating
_, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x)
prediction = get_predictions(A3)[0]

print(f"Prediction: {convert_to_letter(prediction)}")
print(f"Actual:     {convert_to_letter(true_label)}")

plt.imshow(x.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {convert_to_letter(prediction)} | Actual: {convert_to_letter(true_label)}")
plt.savefig("prediction_numpy.png")
print("Saved plot as prediction_numpy.png")

