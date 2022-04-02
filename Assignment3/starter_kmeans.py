import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter

# Loading data
#data = np.load('data2D.npy')
data = np.load('data100D.npy')
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


# Distance function for K-means
def distance_func(X, mu):
    """ Inputs:
          X: is an NxD matrix (N observations and D dimensions)
          mu: is an KxD matrix (K means and D dimensions)
          
        Output:
          pair_dist: is the squared pairwise distance matrix (NxK)
    """

    broad_X = tf.expand_dims(X, axis=1)
    broad_mu = tf.expand_dims(mu, axis=0)

    return tf.reduce_sum(
        tf.square(tf.subtract(broad_X, broad_mu)),
        axis=-1
    )


def train (K=3, epochs=300):
    # Initialize clusters
    mu = tf.Variable(
        initial_value=tf.random.normal(shape=[K, dim]),
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
            # Calculate distances and loss
            distances = distance_func(tf.convert_to_tensor(data, dtype=tf.float32), mu)
            loss = tf.reduce_sum(tf.reduce_min(distances, axis=1))

        # Perform gradient descent
        grads = tape.gradient(loss, [mu])
        opt.apply_gradients(zip(grads, [mu]))

        # Record the loss value
        losses.append(loss.numpy())

    # Validation loss
    val_distances = distance_func(tf.convert_to_tensor(val_data, dtype=tf.float32), mu)
    val_loss = tf.reduce_sum(tf.reduce_min(val_distances, axis=1))
    print("Validation loss = {:.3f}".format(val_loss))

    # Retrieve the assignment clusters for each data point
    # Print the percentage of points in each cluter
    assignments = tf.argmin(distances, 1)
    counter = Counter(assignments.numpy())
    
    '''
    # Plot the training curve and scatter plot
    plt.figure(figsize=(10, 4))
    plt.suptitle('K-Means Algorithm\n(K = {})'.format(K))

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
    ''''
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
    train(K=30)