from src.data.load_dataset import load_and_preprocess_data
#from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_Loan_Amount_distribution
#from src.visualization.visualize import plot_confusion_matrix
from src.feature.build_features import process_data,split_data
from src.model.train_model import train_model, create_model,compile_model
from src.model.predict_model import evaluate_model,predict_model
from src.visualization.visualize import plot_loss_curves,print_classification_report
import numpy as np

import tensorflow as tf


if __name__ == "__main__":
    # Load data
    data_path = "data/employee_attrition.csv"
    df = load_and_preprocess_data(data_path)
        
    # process data 
    X_scaled,Y = process_data(df)
    x_train, x_test, y_train, y_test = split_data(X_scaled, Y)
    
    # Create  & complie model
    input_shape = (x_train.shape[1],)
    num_classes = len(Y.unique())
    model = create_model(input_shape, num_classes)
    model = compile_model(model)

    #Train model
    history = train_model(model, x_train, y_train)
   
   # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Plot loss curves
    plot_loss_curves(history)

     # Get predictions and print classification report
    y_preds=predict_model(model,x_test)
    y_preds = np.round(y_preds).astype(int)
    
    print_classification_report(y_test, y_preds)


   