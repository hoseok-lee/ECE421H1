import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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



def hypothesis(w, b, x):
    return tf.math.sigmoid(tf.matmul(x, w) + b)

def buildGraph():
    # Initialize TensorFlow Graph object
    graph = tf.Graph()

    # Set random seed for reproducible results
    np.random.seed(421)
    tf.set_random_seed(421)

    with graph.as_default():
        # Initialize weight and bias tensors
        # Inefficient to import all data to check size for weights, we already know the size will be
        # 28 * 28 = 784 by 1 dimensional tensor, and bias will be a 1 by 1 tensor
        w = tf.Variable(tf.truncated_normal(shape=(28 * 28, 1), mean=0.0, stddev=0.5, dtype=tf.float32))
        b = tf.Variable(tf.zeros(1))

        # Placeholders for data, labels, and reg
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.uint8)
        reg = tf.placeholder(tf.float32)

        # Calculate loss and optimizer
        y_hat = hypothesis(w, b, x)
        loss = tf.losses.sigmoid_cross_entropy(y, y_hat)
        # Hardcode learning rate alpha = 0.001
        optimizer = tf.train.AdamOptimizer(epsilon=1e-4, learning_rate=1e-3).minimize(loss)
        _, accuracy = tf.metrics.accuracy(y, y_hat)

    return w, b, x, y, reg, y_hat, loss, optimizer, accuracy, graph

def train(w, b, x, y, loss, optimizer, accuracy, graph, batch_size, epochs, reg):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # Set random seed for reproducible results
    np.random.seed(421)
    tf.set_random_seed(421)

    # Flatten all input data into a 1-dimensional vector
    # 28 x 28 tensor becomes a 784-length vector
    trainData = trainData.reshape(len(trainData), -1)
    validData = validData.reshape(len(validData), -1)
    testData = testData.reshape(len(testData), -1)

    d = len(trainData[0])
    N = len(trainTarget)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        # For plotting
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []

        # Calculate number of batches
        batches = int(N / batch_size)

        for epoch in range(epochs):
            # Notify every 100 epochs
            if (epoch % 100) == 0:
                print("Running epoch {}...".format(epoch))

            # Shuffle dataset
            indices = np.arange(N)
            np.random.shuffle(indices)
            trainData, trainTarget = trainData[indices], trainTarget[indices]

            for batch in range(batches):
                batch_index = batch * batch_size

                # Run optimizer
                sess.run(optimizer,
                    feed_dict={
                        x : trainData[batch_index:(batch_index + batch_size)], 
                        y : trainTarget[batch_index:(batch_index + batch_size)],
                        reg : 0
                    }
                )
            
            # Track training and validation loss for plotting
            train_loss.append(sess.run(loss, 
                feed_dict={
                    x : trainData, 
                    y : trainTarget
                }
            ))
            valid_loss.append(sess.run(loss, 
                feed_dict={
                    x : validData, 
                    y : validTarget
                }
            ))
            # And accuracy
            train_acc.append(sess.run(accuracy, 
                feed_dict={
                    x : trainData, 
                    y : trainTarget
                }
            ))
            valid_acc.append(sess.run(accuracy, 
                feed_dict={
                    x : validData, 
                    y : validTarget
                }
            ))

        test_acc = sess.run(accuracy, 
            feed_dict={
                x : testData, 
                y : testTarget
            }
        )

    # Plot the training curves
    plt.figure(figsize=(10, 4))
    plt.suptitle('Stochastic Gradient Descent Loss and Accuracy\n(alpha = {}, epochs = {}, batch size = {})'\
        .format(1e-3, epochs, batch_size))

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

    # Print final accuracies
    print("Training accuracy = {}".format(train_acc[-1]))
    print("Validation accuracy = {}".format(valid_acc[-1]))
    print("Testing accuracy = {}".format(test_acc))


w, b, x, y, reg, y_hat, loss, optimizer, accuracy, graph = buildGraph()
train(
    w=w,
    b=b,
    x=x,
    y=y,
    loss=loss,
    optimizer=optimizer,
    accuracy=accuracy,
    graph=graph,

    batch_size=500,
    epochs=700,
    reg=reg
)