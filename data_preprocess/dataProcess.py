# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.metrics



def featureEncoder(df):
    """
    Transform non-numerical features into values between 0 and n_classes-1
    """
    res = df.copy()
    le = LabelEncoder()
    for col in res.columns:
        if res[col].dtype == np.object:
            mask = ~res[col].isnull()
            res[col][mask] = le.fit_transform(res[col][mask])
            
    return res

    


def main():
    data = pd.read_csv('adult.data.txt',names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],sep=r'\s*,\s*',engine='python',na_values="?")

    
    encoded_data = featureEncoder(data)
    encoded_data = encoded_data.fillna(method='ffill')
    
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(encoded_data[[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country"]], encoded_data["Target"],train_size=0.70)
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # train with a model
    cls = sklearn.linear_model.LogisticRegression()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print (cm)
    print ("F1 score: {0:.4f}".format(sklearn.metrics.f1_score(y_test, y_pred)))
    
if __name__ == '__main__':   
    main()



