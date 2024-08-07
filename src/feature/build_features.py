import pandas as pd
import logging
#to scale the data using z-score
from sklearn.preprocessing import StandardScaler
#to split the dataset
from sklearn.model_selection import train_test_split
# 
def process_data(df):
    try:
       
        #Separating target variable and other variables
        Y= df.Attrition
        X= df.drop(columns = ['Attrition'])

        #Scaling the data
        sc=StandardScaler()
        X_scaled=sc.fit_transform(X)
        X_scaled=pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled,Y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def split_data(X, Y, test_size=0.2, random_state=1):
    try:
        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state,stratify=Y)

        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))