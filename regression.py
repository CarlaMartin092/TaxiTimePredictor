import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import feature_engineering

def preprocess_for_reg(data, train_size = 0.7):
    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    X_train, y_train, X_test, y_test = train_test_split(data_scaled.drop(["y"]), data_scaled["y"], train_size = train_size)
    return(X_train, y_train, X_test, y_test, scaler)

def fit_regression(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return(reg)

def assess_mae(data, train_size = 0.7):
    X_train, y_train, X_test, y_test, scaler = preprocess_for_reg(data, train_size=0.7)
    fitted_reg = fit_regression(X_train, y_train)
    y_pred = fitted_reg.predict(X_test)
    return(mean_absolute_error(y_pred, y_test))

def predict_regression(reg, x):

    return(reg.predict(x))



