import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(w, b, x):
    return sigmoid(np.matmul(x, w) + b)

def loss(w, b, x, y, reg):
    y_hat = hypothesis(w, b, x)

    N = len(y)

    L_CE = (1 / N) * np.sum(-(y * np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
    L_w = (reg / 2) * np.power(np.linalg.norm(w), 2)
    
    L = L_CE + L_w

    return L

def grad_loss(w, b, x, y, reg):
    y_hat = hypothesis(w, b, x)

    N = len(y)

    grad_w = (1 / N) * np.matmul(x.T, (y_hat - y)) + (reg * w)
    grad_b = (1 / N) * np.sum(y_hat - y) + (reg  * b)

    return grad_w, grad_b

# Function for calculating accuracy
def accuracy(w, b, x, y):
    y_hat = hypothesis(w, b, x)

    N = len(y)

    # Assuming 0.5 as the probability separation of classes, 
    # convert probabilities into predictions and calculate accuracy
    return np.count_nonzero((y_hat > 0.5) == y) / N

def grad_descent(w, b, X, Y, alpha, epochs, reg, error_tol):

    # Unpack data
    train_x, valid_x, test_x = X
    train_y, valid_y, test_y = Y

    # For plotting
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    # For error tolerance convergence
    w_old = w
    for epoch in range(epochs):
        # Notify every 100 epochs
        if (epoch % 100) == 0:
            print("Running epoch {}...".format(epoch))

        # Retrieve gradients
        grad_w, grad_b = grad_loss(w, b, train_x, train_y, reg)
        
        # Update weight and bias vectors according to gradient loss
        # Incorporate learning rate
        w = w - (alpha * grad_w)
        b = b - (alpha * grad_b)

        # Track training and validation loss for plotting
        train_loss.append(loss(w, b, train_x, train_y, reg))
        valid_loss.append(loss(w, b, valid_x, valid_y, reg))
        # And accuracy
        train_acc.append(accuracy(w, b, train_x, train_y))
        valid_acc.append(accuracy(w, b, valid_x, valid_y))

        # Check for error tolerance, break if reached assumed convergence
        if np.linalg.norm(w - w_old) < error_tol:
            print("Error tolerance reached.")
            break

    # Plot the training curves
    plt.figure(figsize=(10, 4))
    plt.suptitle('Logistic Regression Loss and Accuracy\n(alpha = {}, epochs = {}, reg = {}, error_tol = {})'\
        .format(alpha, epochs, reg, error_tol))

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

    return w, b

def train(alpha, epochs, reg, error_tol=1e-7):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # Flatten all input data into a 1-dimensional vector
    # 28 x 28 tensor becomes a 784-length vector
    trainData = trainData.reshape(len(trainData), -1)
    validData = validData.reshape(len(validData), -1)
    testData = testData.reshape(len(testData), -1)

    d = len(trainData[0])
    N = len(trainTarget)

    # Initialize weight and bias vectors
    w = np.zeros((d, 1))
    b = np.zeros((1, 1))

    # Perform gradient descent
    w, b = grad_descent(
        w=w,
        b=b,
        X=[trainData, validData, testData],
        Y=[trainTarget, validTarget, testTarget],
        alpha=alpha,
        epochs=epochs,
        reg=reg,
        error_tol=error_tol
    )

    # Print final accuracies
    print("Training accuracy = {}".format(accuracy(w, b, trainData, trainTarget)))
    print("Validation accuracy = {}".format(accuracy(w, b, validData, validTarget)))
    print("Testing accuracy = {}".format(accuracy(w, b, testData, testTarget)))

train(
    alpha=5e-3,
    epochs=5000,
    reg=0.5
)