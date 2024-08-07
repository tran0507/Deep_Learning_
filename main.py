from src.data.load_dataset import load_and_preprocess_data
#from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_Loan_Amount_distribution
#from src.visualization.visualize import plot_confusion_matrix
from src.feature.build_features import process_data,split_data
from src.model.train_model import train_model, create_compile_model,train_with_history
from src.model.predict_model import evaluate_model,predict_model
from src.visualization.visualize import plot_training_curves,plot_learning_rate_vs_loss
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Load data
    data_path = "data/employee_attrition.csv"
    df = load_and_preprocess_data(data_path)
        
    # process data 
    X_scaled,Y = process_data(df)
    x_train, x_test, y_train, y_test = split_data(X_scaled, Y)
    
    # Create  & complie model

    model = create_compile_model(layers=[1])

    #Train model 
    print("\nThe first model with 1 layer")
    history,model= train_model(model, x_train, y_train, epochs=5)
    train_accuracy= model.evaluate(x_train,y_train) #evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Initial model train accuracy:", train_accuracy)
  
    # Training longer
    print("\nModel with more epochs")
    train_model(model, x_train, y_train, epochs=100)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)
    print("Longer training train accuracy:", train_accuracy)
    
   # Model with an extra layer

    print("\nModel with an extra layer")
    model= create_compile_model(layers=[1, 1])
    train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)
    print("Extra layer model train accuracy:", train_accuracy)
    

    # Model with 2 neurons in the hidden layer
    print("\nModel with 2 neurons in the hidden layer")
    model = create_compile_model(layers=[2, 1])
    train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("2 neurons model train accuracy:", train_accuracy)
    

    # Model with another extra layer
    print("\nModel with 3 layers")
    model = create_compile_model(layers=[1, 1, 1])
    train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Another extra layer model train accuracy:", train_accuracy)
   
    # Model with modified learning rate
    print("\nModel with different training rate")
    model = create_compile_model(layers=[1, 1], learning_rate=0.0009)
    train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)
    print("Modified learning rate model train accuracy:", train_accuracy)
   
    # Model with learning rate scheduler
    print("\nModel with learning rate scheduler")
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * 0.9**(epoch/3)
        )
    model=create_compile_model(layers=[1,1],learning_rate=0.0009)
    history=train_model(model, x_train, y_train, epochs=100,callbacks=[lr_scheduler])
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)
   
 
    #plot learning & loss rate 
    #plot_learning_rate_vs_loss(history)

    # Final model with sigmoid activation
    
    print("Model with activation=sigmoid")
       
    model = create_compile_model(layers=[1, 1], learning_rate=0.0009, activation='sigmoid')
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Sigmoid activation model train accuracy:", train_accuracy)
  
    # Predictions and evaluation
    y_preds = predict_model(model, x_test)
    print("Predictions on test set:", y_preds[:3])
    print("Accuracy on test set:", accuracy_score(y_test, y_preds))
  
    #plot_training_curves(history)

   