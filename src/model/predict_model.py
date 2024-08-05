import pandas as pd
import logging
import tensorflow as tf
from sklearn.metrics import accuracy_score
# 
def predict_evaluate(model, x_test,y_test):
    try:
       
        y_preds = tf.round(model.predict(x_test))
        accuracy_score= accuracy_score(y_test, y_preds))
    
        return accuracy_score 
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))