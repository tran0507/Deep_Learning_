from src.data.load_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_Loan_Amount_distribution
#from src.visualization.visualize import plot_confusion_matrix
from src.feature.build_features import create_features
from src.model.train_model import train_model
from src.model.predict_model import predict_model


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/credit.csv"
    df = load_and_preprocess_data(data_path)
        
    #show distribution of Loan amount
    #plot_Loan_Amount_distribution(df)

    # Create dummy variables and separate features and target
    x, y = create_features(df)

    # Train Random Forest Classification model
    model, X_test, y_test = train_model(x, y)
    #print(f"Train accuracy_score: {train_accuracy_score}")
    #print(f"Train confusion_matrix: {train_confusion_matrix}")

     #Predict & Evaluate the model
    accuracy,confusion_mat = predict_model(model, X_test, y_test)
    print(f"Test accuracy_score: {accuracy}")
    print(f"Test confusion_matrix: {confusion_mat}")

    #plot_confusion_matrix(y_test,y_pred)