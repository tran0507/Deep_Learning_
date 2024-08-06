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


def create_model(input_shape, num_classes):
    try: 
        # Define the model architecture
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        return model

    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def compile_model(model,learning_rate=0.009):
    try: 
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return model

    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def train_model(model, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    try:
       
        print("\n Start train model")
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
        # Save the trained model
        with open('model/deeplearning_tf.pkl', 'wb') as f:
            pickle.dump(model, f)
        return history

    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))