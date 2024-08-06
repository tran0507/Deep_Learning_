import pandas as pd
import logging

from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# plot loss curve
def plot_loss_curves(history):
    try: 
        # Plot the loss curves
        pd.DataFrame(history.history).plot()
        plt.title("Model training curves")
        plt.show()

    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def print_classification_report(y_test, y_preds):
    try: 
        # Print the classification report
        report = classification_report(y_test, y_preds)
        print(report)
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))