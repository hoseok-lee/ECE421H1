import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def load_data():
    with np.load("notMNIST.npz") as data:
        data, targets = data["images"], data["labels"]
        
        np.random.seed(521)
        rand_idx = np.arange(len(data))
        np.random.shuffle(rand_idx)
        
        data = data[rand_idx] / 255.0
        targets = targets[rand_idx].astype(int)
        
        train_data, train_target = data[:10000], targets[:10000]
        valid_data, valid_target = data[10000:16000], targets[10000:16000]
        test_data, test_target = data[16000:], targets[16000:]
    return train_data, valid_data, test_data, train_target, valid_target, test_target


def convert_onehot(train_target, valid_target, test_target):
    new_train = np.zeros((train_target.shape[0], 10))
    new_valid = np.zeros((valid_target.shape[0], 10))
    new_test = np.zeros((test_target.shape[0], 10))

    for item in range(0, train_target.shape[0]):
        new_train[item][train_target[item]] = 1
    for item in range(0, valid_target.shape[0]):
        new_valid[item][valid_target[item]] = 1
    for item in range(0, test_target.shape[0]):
        new_test[item][test_target[item]] = 1
    return new_train, new_valid, new_test



def shuffle(data, target):
    np.random.seed(421)
    rand_idx = np.random.permutation(len(data))
    return data[rand_idx], target[rand_idx]

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def compute_layer(x, w, b):
    return np.matmul(x, w) + b

def average_ce(target, prediction):
    return -np.mean(np.sum((target * np.log(prediction)), axis=1))

def grad_ce(target, logits):
    return np.mean(softmax(logits) - target, axis=0, keepdims=True)

def forward_prop(x, w_h, b_h, w_o, b_o):
    # Hidden layer
    z_h = compute_layer(x, w_h, b_h)
    a_h = relu(z_h)

    # Output layer
    z_o = compute_layer(a_h, w_o, b_o)
    a_o = softmax(z_o)

    return z_h, z_o, a_h, a_o

# Backward propagation gradients
def backward_prop(target, prediction, x, z_h, a_h, w_o):
    total_ce = prediction - target

    # Hidden layer
    dw_h = np.matmul(x.T, np.where(z_h > 0, 1, 0) * np.matmul(total_ce, w_o.T))
    db_h = np.matmul(np.ones((1, target.shape[0])), np.where(z_h > 0, 1, 0) * np.matmul(total_ce, w_o.T))

    # Output layer
    dw_o = np.matmul(a_h.T, total_ce)
    db_o = np.matmul(np.ones((1, target.shape[0])), total_ce)

    return dw_h, db_h, dw_o, db_o

# Function for calculating accuracy
def accuracy(target, prediction):
    # Conver one-hot encoding to class
    target_class = np.argmax(target, axis=1)
    prediction_class = np.argmax(prediction, axis=1)

    N = len(target)

    # Compare the matrix of predictions to targets
    return np.count_nonzero(target_class == prediction_class) / N

def grad_descent(w, b, v, X, Y, alpha, epochs, gamma):
    # Unpack weights, biases, and velocities
    w_h, w_o = w
    b_h, b_o = b
    v_w_h, v_w_o, v_b_h, v_b_o = v

    # Unpack data
    train_x, valid_x, test_x = X
    train_y, valid_y, test_y = Y

    # For plotting
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for epoch in range(epochs):
        # Notify every 10 epochs
        if (epoch % 50) == 0:
            print("Running epoch {}...".format(epoch))

        # Forward propagation
        z_h, _, a_h, a_o = forward_prop(train_x, w_h, b_h, w_o, b_o)
        train_loss.append(average_ce(train_y, a_o))
        train_acc.append(accuracy(train_y, a_o))

        _, _, _, valid_pred = forward_prop(valid_x, w_h, b_h, w_o, b_o)
        valid_loss.append(average_ce(valid_y, valid_pred))
        valid_acc.append(accuracy(valid_y, valid_pred))

        # Backward propagation
        dw_h, db_h, dw_o, db_o = backward_prop(train_y, a_o, train_x, z_h, a_h, w_o)

        # Update weight and bias vectors according to gradient loss
        # Incorporate learning rate and momentum
        v_w_h = (gamma * v_w_h) + (alpha * dw_h)
        w_h = w_h - v_w_h
        v_b_h = (gamma * v_b_h) + (alpha * db_h)
        b_h = b_h - v_b_h

        v_w_o = (gamma * v_w_o) + (alpha * dw_o)
        w_o = w_o - v_w_o
        v_b_o = (gamma * v_b_o) + (alpha * db_o)
        b_o = b_o - v_b_o

    # Plot the training curves
    plt.figure(figsize=(10, 4))
    plt.suptitle('Logistic Regression Loss and Accuracy\n(alpha = {}, epochs = {}, gamma = {})'\
        .format(alpha, epochs, gamma))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(valid_loss, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(valid_acc, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return [w_h, w_o], [b_h, b_o]

def train(alpha, epochs, gamma):
    train_data, valid_data, test_data, train_target, valid_target, test_target = load_data()

    # Flatten all input data into a 1-dimensional vector
    # 28 x 28 tensor becomes a 784-length vector
    train_data = train_data.reshape(len(train_data), -1)
    valid_data = valid_data.reshape(len(valid_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    # Convert to one-hot encoding for targets
    train_target, valid_target, test_target = convert_onehot(train_target, valid_target, test_target)

    F = len(train_data[0])
    N = len(train_target)
    H = 1000
    K = 10

    # Xavier initialization
    mean = 0
    v_init = 1e-5

    # Hidden layer
    var_h = 2 / (F + H)
    w_h = np.random.normal(mean, var_h, (F, H))
    b_h = np.zeros((1, H))
    v_w_h = np.full((F, H), v_init)
    v_b_h = np.zeros((1, H))

    # Output layer
    var_o = 2 / (H + K)
    w_o = np.random.normal(mean, var_o, (H, K))
    b_o = np.zeros((1, K))
    v_w_o = np.full((H, K), v_init)
    v_b_o = np.zeros((1, K))

    # Perform gradient descent with momentum
    w, b = grad_descent(
        w=[w_h, w_o],
        b=[b_h, b_o],
        v=[v_w_h, v_w_o, v_b_h, v_b_o],
        X=[train_data, valid_data, test_data],
        Y=[train_target, valid_target, test_target], 
        alpha=alpha, 
        epochs=epochs, 
        gamma=gamma
    )

    # Unpack weights and biases
    w_h, w_o = w
    b_h, b_o = b
    
    _, _, _, train_pred = forward_prop(train_data, w_h, b_h, w_o, b_o)
    _, _, _, valid_pred = forward_prop(valid_data, w_h, b_h, w_o, b_o)
    _, _, _, test_pred = forward_prop(test_data, w_h, b_h, w_o, b_o)

    # Print final accuracies
    print("Training accuracy = {}".format(accuracy(train_target, train_pred)))
    print("Validation accuracy = {}".format(accuracy(valid_target, valid_pred)))
    print("Testing accuracy = {}".format(accuracy(test_target, test_pred)))

train(
    alpha=1e-5,
    epochs=200,
    gamma=0.99
)