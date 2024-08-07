import pandas as pd
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_learning_rate_vs_loss(history):
    try: 
        """
        Plot the learning rate versus the loss.

        :param history: History - Training history
        """
        
        lrs = 1e-5 * (10 ** (np.arange(100) / 20))
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"])  # We want the x-axis (learning rate) to be log scale
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
    


def plot_training_curves(history):
    try: 
        """
        Plot the loss and accuracy curves for the model training.

        :param history: History - Training history
        """
        pd.DataFrame(history.history).plot()
        plt.title("Model training curves")
        plt.show()
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))


