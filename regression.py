import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import feature_engineering as ft
from scipy import stats
import os
import metrics
import matplotlib.pyplot as plt

class Regression():
    """
    Multilinear regression class. Scales input data and performs multilinear regression with target variable "y". Allows to retrieve \
    the regression coefficients.
    """
    def __init__(self):
        """
        Initializes scalers and sklearn LinearRegression object.
        """
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()
        self.coef_ = []
        self.p = []
        self.r2 = 0.0
        self.reg = LinearRegression()
    
    def preprocess(self, data, train_size = 0.7, random_state = None):
        """
        Rescales the input data by subtracting the mean and dividing by the standard deviation of each variable. Splits the data into 
        training and testing data.

        Arguments:
        - data: pandas dataframe, contains the independent variables and the dependent variable called "y"
        - train_size: float, proportion of the data that is used for training
        - random_state: int or None, sets a random seed
        """
        X = data.loc[:, data.columns != "actual_taxi_out_sec"]
        y = data.loc[:,"actual_taxi_out_sec"]
        self.x_scaler.fit(X)
        self.y_scaler.fit(np.array(y).reshape(-1, 1))
        X_rescaled = self.x_scaler.transform(X)
        y_rescaled = self.y_scaler.transform(np.array(y).reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X_rescaled, y_rescaled, train_size = train_size, random_state = random_state)
        return(X_train, X_test, y_train, y_test)

    def fit(self, X_train, y_train):
        self.reg.fit(X_train, y_train)
        self.coef_ = self.reg.coef_

        sse = np.sum((self.reg.predict(X_train) - y_train) ** 2, axis=0) / float(X_train.shape[0] - X_train.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X_train.T, X_train))))
                                                    for i in range(sse.shape[0])])
        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y_train.shape[0] - X_train.shape[1]))
        return self

    def predict(self, x):
        y_pred = self.reg.predict(x.reshape(1, -1))
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred)
        return(y_pred_rescaled)
    
    def evaluate(self, X_test, y_test, interval = 2.0, plot_prediction = True):
        y_pred = self.reg.predict(X_test)
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = self.y_scaler.inverse_transform(y_test.reshape(-1, 1))

        if(plot_prediction):
            plt.plot(y_test_rescaled)
            plt.plot(y_pred_rescaled)
            plt.legend(["Ground Truth", "Prediction"])
            plt.show()

        return(r2_score(y_test_rescaled, y_pred_rescaled), mean_absolute_error(y_test_rescaled, y_pred_rescaled), metrics.accuracy(y_test_rescaled, y_pred_rescaled, interval = interval))
    
    def get_coef_(self):
        return(self.coef_)

    def get_p_values(self):
        return self.p

def run_regression(mode = "train", run_feature_engineering = False, include_dummies = False, train_size = 0.7, random_state = None):

    if (os.path.exists("../data/taxitime_train_variables.csv") and not run_feature_engineering):
        taxitime_data = pd.read_csv("../data/taxitime_train_variables.csv")
        taxitime_data = taxitime_data.drop(["Unnamed: 0"], axis = 1)

    else: 
        taxitime_data = ft.feature_engineering(include_dummies = include_dummies)

    reg = Regression()
    taxitime_data = taxitime_data.dropna(subset = ["windGust"])
    #print(np.sum(pd.isna(taxitime_data)))
    X_train, _, y_train, _ = reg.preprocess(taxitime_data, train_size = 0.99, random_state = random_state)
    reg = reg.fit(X_train, y_train)
    print("Coefficients: ", reg.get_coef_())
    print("P-Values: ", reg.get_p_values())

    if(mode == "test"):
        test_data = pd.read_csv("../data/taxitime_test_variables.csv")
        test_data = test_data.drop(["Unnamed: 0"], axis = 1)
        test_data = test_data.dropna(subset = ["windGust"])
        _, X_test, _, y_test = reg.preprocess(test_data, train_size = 0.01, random_state = random_state)
        r2, mae, accuracy = reg.evaluate(X_test, y_test, interval = 2.0)
        print("R2 Score: ", r2)
        print("MAE: ", mae)
        print("Accuracy: ", accuracy)
    #print("Prediction for ", reg.x_scaler.inverse_transform(X_test[0,:]).T, ": ",  reg.predict(X_test[0,:]), "\True value: ", reg.y_scaler.inverse_transform(y_test[0,:]).T)

run_regression(mode = "test")


