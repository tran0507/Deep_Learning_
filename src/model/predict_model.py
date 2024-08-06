import pandas as pd
import logging
import tensorflow as tf
from sklearn.metrics import accuracy_score
# 
def evaluate_model(model, x_test,y_test):
    try:
        print("\n Start evaluate model")
              
        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        return test_loss, test_accuracy
    
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def predict_model(model,x_test):
    try: 
        # Get predictions and print classification report
        print("Start predict model")
        y_preds = tf.round(model.predict(x_test))

        return y_preds
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))