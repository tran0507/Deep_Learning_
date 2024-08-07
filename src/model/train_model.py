import pandas as pd
import logging

#to scale the data using z-score
from sklearn.preprocessing import StandardScaler

#to split the dataset
from sklearn.model_selection import train_test_split

#Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#to ignore warnings
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np


def create_compile_model(layers,learning_rate=0.001,activation=None):
    try: 

        tf.keras.utils.set_random_seed(42)
        model=None
        model = tf.keras.Sequential()        
        for i, layer_size in enumerate(layers): 
            if i==len(layers)-1: 
                model.add(tf.keras.layers.Dense(layer_size))
            else: 
                model.add(tf.keras.layers.Dense(layer_size,activation=activation ))
         
        
        # Compile the model
        model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                    metrics=['accuracy'])
    
        return model

    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))


def train_model(model, x_train, y_train, epochs=50, callbacks=None):
    try:       
 
        history=model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=callbacks)

        # Save the trained model
        with open('model/deeplearning_tf.pkl', 'wb') as f:
            pickle.dump(model, f)
        return history, model

    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
