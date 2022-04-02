from tkinter.tix import TCL_FILE_EVENTS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter
import starter_kmeans

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = True

if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]


def distance_func(X, mu):
    """ Inputs:
            X: is an NxD matrix (N observations and D dimensions)
            mu: is an KxD matrix (K means and D dimensions)
        Outputs
            pair_dist: is the pairwise distance matrix (NxK)
    """
    
    # TODO


def log_gauss_pdf(X, mu, sigma):
    """ Inputs: 
            X: N X D
            mu: K X D
            sigma: K X 1

        Outputs:
            log Gaussian PDF (N X K)
    """

    # sigma_T: 1 X K
    sigma_T = tf.transpose(sigma)
    
    distance = starter_kmeans.distance_func(X, mu)
    return tf.subtract(
        tf.multiply(-dim, tf.math.log(tf.sqrt(2 * np.pi) * sigma_T)),
        tf.divide(distance, (2 * tf.square(sigma_T)))
    )


def log_posterior(log_PDF, log_pi):
    """ Inputs:
            log_PDF: log Gaussian PDF N X K
            log_pi: K X 1

        Outputs
            log_post: N X K
    """
    
    # Avoid computational redundancy
    sum_log_PDF_pi = tf.add(log_PDF, tf.transpose(log_pi))
    return tf.subtract(
        sum_log_PDF_pi,
        hlp.reduce_logsumexp(sum_log_PDF_pi, keepdims=True)
    )


def train (K=3, epochs=300):
    # Initialize variables
    mu = tf.Variable(
        initial_value=tf.random.normal(shape=[K, dim]),
        trainable=True,
        dtype=tf.float32
    )
    phi = tf.Variable(
        initial_value=tf.random.normal(shape=[K, 1]),
        trainable=True,
        dtype=tf.float32
    )
    psi = tf.Variable(
        initial_value=tf.random.normal(shape=[K, 1]),
        trainable=True,
        dtype=tf.float32
    )

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.99,
        epsilon=1e-5
    )

    losses = []
    for _ in range(epochs):
        # Compute loss function
        with tf.GradientTape() as tape:
            # Adjust constraints
            sigma_squared = tf.exp(phi)
            log_pi = hlp.logsoftmax(psi)

            # Calculate log posterior probability
            log_PDF = log_gauss_pdf(
                X=tf.convert_to_tensor(data, dtype=tf.float32), 
                mu=mu, 
                sigma=tf.sqrt(sigma_squared)
            )
            log_post = log_posterior(log_PDF, log_pi)

            # Calculate loss
            loss = -tf.reduce_sum(
                hlp.reduce_logsumexp(
                    tf.add(log_PDF, tf.transpose(log_pi)),
                    keepdims=True
                )
            )

        # Perform gradient descent
        grads = tape.gradient(loss, [mu, phi, psi])
        opt.apply_gradients(zip(grads, [mu, phi, psi]))

        # Record the loss value
        losses.append(loss.numpy())
    
    # Validation loss
    val_log_PDF = log_gauss_pdf(
        X=tf.convert_to_tensor(val_data, dtype=tf.float32), 
        mu=mu, 
        sigma=tf.sqrt(sigma_squared)
    )
    val_loss = -tf.reduce_sum(
        hlp.reduce_logsumexp(
            tf.add(val_log_PDF, tf.transpose(log_pi)),
            keepdims=True
        )
    )
    print("Validation loss = {:.3f}".format(val_loss.numpy()))

    # Plot the training curve and scatter plot
    plt.figure(figsize=(10, 4))
    plt.suptitle('Gaussian Mixture Model Expectation-Maximization Algorithm\n(K = {})'.format(K))

    # Cluster assignment
    assignments = tf.argmax(log_post, axis=1)
    counter = Counter(assignments.numpy())
    '''
    # Scatter plot
    plt.subplot(1, 2, 1)
    '''
    percentages = []
    for cluster in range(K):
        # Edge case where no point is assigned to cluster
        cluster_population = 0
        if cluster in counter.keys():
            cluster_population = counter[cluster]
        percentage = cluster_population / len(data)
        percentages.append(percentage)
    print("Maximum: {:.2%}".format(max(percentages)))
    print("Median: {:.2%}".format(np.median(percentages)))
    print("Minimum: {:.2%}".format(min(percentages)))
    '''
        indices = (assignments == cluster)
        plt.scatter(
            data[:, 0][indices], data[:, 1][indices],
            alpha=0.5,
            label='{:.2%}'.format(percentage)
        )
    plt.plot(mu[:, 0], mu[:, 1], 'kx')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    # Training curve
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    '''

if __name__ == "__main__":
    train(K=5)