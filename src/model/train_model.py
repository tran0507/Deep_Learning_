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

# 

def create_complile_model(learning_rate):
    try: 
        #Creates and compiles the TensorFlow model."""
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential([
             tf.keras.layers.Dense(1), 
             tf.keras.layers.Dense(1, activation = 'sigmoid') 
            ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  metrics=['accuracy'])
        return model

    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def train_model(model, x_train,y_train,epochs=50, lr_scheduler=None):
    try:
       
        if lr_scheduler:
            history = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[lr_scheduler])
        else:
            history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
        
        
        # Save the trained model
        with open('model/deeplearning_tf.pkl', 'wb') as f:
            pickle.dump(model, f)
        return history

    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))