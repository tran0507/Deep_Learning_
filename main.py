from src.data.load_dataset import load_and_preprocess_data
#from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_Loan_Amount_distribution
#from src.visualization.visualize import plot_confusion_matrix
from src.feature.build_features import process_data,split_data
from src.model.train_model import train_model, create_compile_model
from src.model.predict_model import evaluate_model,predict_model
from src.visualization.visualize import plot_training_curves,plot_learning_rate_vs_loss
import numpy as np

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


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
    history = train_model(model, x_train, y_train, epochs=5)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Initial model train accuracy:", train_accuracy)
    print("Initial model test accuracy:", test_accuracy)

    # Training longer
    history = train_model(model, x_train, y_train, epochs=100)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Longer training train accuracy:", train_accuracy)
    print("Longer training test accuracy:", test_accuracy)
   
   # Model with an extra layer
    model= create_compile_model(layers=[1, 1])
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Extra layer model train accuracy:", train_accuracy)
    print("Extra layer model test accuracy:", test_accuracy)

    # Model with 2 neurons in the hidden layer
    model = create_compile_model(layers=[2, 1])
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("2 neurons model train accuracy:", train_accuracy)
    print("2 neurons model test accuracy:", test_accuracy)

    # Model with another extra layer
    model = create_compile_model(layers=[1, 1, 1])
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Another extra layer model train accuracy:", train_accuracy)
    print("Another extra layer model test accuracy:", test_accuracy)

    # Model with modified learning rate
    model = create_compile_model(layers=[1, 1], learning_rate=0.0009)
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Modified learning rate model train accuracy:", train_accuracy)
    print("Modified learning rate model test accuracy:", test_accuracy)

    # Model with learning rate scheduler
    model= create_compile_model(layers=[1, 1])
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )
    history = train_model(model, x_train, y_train, epochs=100, callbacks=[lr_scheduler])

    #plot learning & loss rate 
    plot_learning_rate_vs_loss(history)

    # Final model with sigmoid activation
    model = create_compile_model(layers=[1, 1], learning_rate=0.0009, activation='sigmoid')
    history = train_model(model, x_train, y_train, epochs=50)
    train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("Sigmoid activation model train accuracy:", train_accuracy)
    print("Sigmoid activation model test accuracy:", test_accuracy)

    # Predictions and evaluation
    y_preds = predict_model(model, x_test)
    print("Predictions on test set:", y_preds[:3])
    print("Accuracy on test set:", accuracy_score(y_test, y_preds))

    # Plot training curves
    plot_training_curves(history)


   