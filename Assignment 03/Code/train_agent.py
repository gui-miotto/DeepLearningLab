# Third-party packages and modules:
import pickle, os, gzip
import numpy as np
import matplotlib.pyplot as plt
# My packages and modules:
from agent import Agent
import utils

def read_data():
    """Reads the states and actions recorded by drive_manually.py"""
    print("Reading data")
    with gzip.open('./data_from_expert/data_02.pkl.gzip','rb') as f:
        data = pickle.load(f)
    X = utils.vstack(data["state"])
    y = utils.vstack(data["action"])
    return X, y

def preprocess_data(X, y, hist_len, shuffle):
    """ Preprocess states and actions from expert dataset before feeding them to the agent """
    print('Preprocessing states. Shape:', X.shape)
    utils.check_invalid_actions(y)
    y_pp = utils.transl_action_env2agent(y)
    X_pp = utils.preprocess_state(X)
    X_pp, y_pp = utils.stack_history(X_pp, y_pp, hist_len, shuffle=shuffle)
    return X_pp, y_pp

def split_data(X, y, frac = 0.1):
    """ Splits data into training and validation set """
    split = int((1-frac) * len(y))
    X_train, y_train = X[:split], y[:split]
    X_valid, y_valid = X[split:], y[split:]
    return X_train, y_train, X_valid, y_valid

def plot_states(x_pp, X_tr=None, n=3):
    """ Plot some random states before and after preprocessing """
    pick = np.random.randint(0, len(x_pp), n)
    fig, axes = plt.subplots(n, 2, sharex=True, sharey=True, figsize=(20,20))
    for i, p in enumerate(pick):
        if X_tr is not None:
            axes[i,0].imshow(X_tr[p]/255)
        axes[i,1].imshow(np.squeeze(x_pp[p]), cmap='gray')
    fig.tight_layout()
    plt.show()

def plot_action_histogram(actions, title):
    """ Plot the histogram of actions from the expert dataset """
    acts_id = utils.unhot(actions)
    fig, ax = plt.subplots()
    bins = np.arange(-.5, utils.n_actions + .5)
    ax.hist(acts_id, range=(0,6), bins=bins, rwidth=.9)
    ax.set(title=title, xlim=(-.5, utils.n_actions -.5))
    plt.show()

if __name__ == "__main__":
    # Read data:
    X, y = read_data()
    # Preprocess it:
    X_pp, y_pp = preprocess_data(X, y, hist_len=utils.history_length, shuffle=False)
    # Plot action histogram. JUST FOR DEBUGGING.
    if True: plot_action_histogram(y_pp, 'Action distribution BEFORE balancing')   
    # Balance samples. Gets hide of 50% of the most common action (accelerate)
    X_pp, y_pp = utils.balance_actions(X_pp, y_pp, 0.5)
    # Plot action histogram. JUST FOR DEBUGGING.
    if True: plot_action_histogram(y_pp, 'Action distribution AFTER balancing')   
    # Plot some random states before and after preprocessing. JUST FOR DEBUGGING. 
    # Requires to run the above fucntion with hist_len=1, shuffle=False.
    if False: plot_states(X_pp, X)
    # Split data into training and validation:
    X_train, y_train, X_valid, y_valid = split_data(X_pp, y_pp, frac=.1)
    # Create a new agent from scratch:
    agent = Agent.from_scratch(n_channels=utils.history_length)
    # Train it:
    agent.train(X_train, y_train, X_valid, y_valid, n_batches=200000, batch_size=100, lr=5e-4, display_step=100)
    # Save it to file:
    agent.save('saved_models/')
 
