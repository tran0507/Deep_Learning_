from src.data.load_dataset import load_and_preprocess_data
#from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_Loan_Amount_distribution
#from src.visualization.visualize import plot_confusion_matrix
from src.feature.build_features import create_features
from src.model.train_model import train_model, create_complile_model
from src.model.predict_model import predict_evaluate 


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/employee_attrition.csv"
    df = load_and_preprocess_data(data_path)
        
    # Create dummy variables and separate features and target
    x_train,x_test,y_train,y_test = create_features(df)

    # Train  model
    model = create_compile_model(learning_rate=0.0009)

    history = train_model(model, x_train, y_train)

    accuracy_score=evaluate_model(model, x_test, y_test)
    print(accuracy_score)
    


   