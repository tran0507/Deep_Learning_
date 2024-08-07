import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 
def evaluate_model(model, x_train,y_train,x_test,y_test):
    try:
                    
        # Evaluate the model
        train_accuracy = model.evaluate(x_train, y_train)
        test_accuracy = model.evaluate(x_test, y_test)
        return train_accuracy, test_accuracy
    
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def predict_model(model,x_test):
    try: 
        # Get predictions
        y_preds = model.predict(x_test)
        y_preds = tf.round(y_preds)
        return y_preds

    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))