from __future__ import print_function

from itertools import product
import cnn_mnist as cnn
import matplotlib.pyplot as plt
import numpy as np


def try_hyperparameters(learning_rates, kernel_sizes):

    learning_curves = []
    for lr, ks in product(learning_rates, kernel_sizes):
        lc, _ = cnn.train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, ks)
        learning_curves.append(lc)

    return np.array(learning_curves).T

def plot_learning_curves(lcurves, curve_labels, title):
    fig, ax = plt.subplots()
    ax.set(title=title, xlabel='Epoch', ylabel='Accuracy')
    plots = ax.plot(lcurves)
    ax.legend(plots, curve_labels, loc=4)
    fig.savefig(title+'.pdf', format='pdf')
    #plt.show()


if __name__ == "__main__":
    # hyperparameters
    lr = 1e-3
    num_filters = 32
    batch_size = 128
    epochs = 12

    # train and test datasets
    x_train, y_train, x_valid, y_valid, x_test, y_test = cnn.mnist()

    # trying different learning rates
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    kernel_sizes = [3]
    lcs = try_hyperparameters(learning_rates, kernel_sizes)
    plot_learning_curves(lcs, learning_rates, 'Learning rates')

    # trying different filter sizes
    learning_rates = [0.001]
    kernel_sizes = [1, 3, 5, 7]
    lcs = try_hyperparameters(learning_rates, kernel_sizes)
    plot_learning_curves(lcs, kernel_sizes, 'kernel sizes')



