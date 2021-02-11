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
    Multilinear regression class. Scales input data and performs multilinear regression with target variable "actual_taxi_out_sec". Allows to retrieve \
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
        """
        Fits the Regression object to the training data. Computes p-values.

        Arguments:
        - X_train: numpy.ndarray, independent variables
        - y_train: numpy.ndarray, dependent variable

        Returns:
        - self: Regression object
        """
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
        """
        Predicts the value of y based on an observation x.

        Arguments:
        - x: numpy.ndarray, independent variables.

        Returns:
        - y_pred: rescaled predictions.
        """

        y_pred = self.reg.predict(x.reshape(1, -1))
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred)
        return(y_pred_rescaled)
    
    def evaluate(self, X_test, y_test, interval = 2.0, plot_prediction = True):
        """
        Evaluates the predictions on the test set based on 3 metrics: R2, mean absolute error and a metric we defined as accuracy, which
        corresponds to the percentage of prediction that are within a 2 minutes (default interval) interval from the true value. 
        See metrics.accuracy for more details.

        Arguments:
        - X_test: numpy.ndarray, independent variables
        - y_test: numpy.ndarray, dependent variables
        - interval: float, number of minutes to build the accuracy interval
        - plot_prediction: bool, if true, saves prediction plots

        Returns:
        - r2: r2 score of the prediction
        - mae: mean absolute error of the prediction
        - accuracy: accuracy measure defined in metrics.accuracy
        """

        y_pred = self.reg.predict(X_test)
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = self.y_scaler.inverse_transform(y_test.reshape(-1, 1))

        if(plot_prediction):
            plt.plot(y_test_rescaled)
            plt.plot(y_pred_rescaled)
            plt.legend(["Ground Truth", "Prediction"], loc = "upper right")
            plt.savefig("../data/regression_predictions.png")

        return(r2_score(y_test_rescaled, y_pred_rescaled), mean_absolute_error(y_test_rescaled, y_pred_rescaled), metrics.accuracy(y_test_rescaled, y_pred_rescaled, interval = interval))
    
    def get_coef_(self):
        """
        Returns the coefficients of the regression. 
        """
        return(self.coef_)

    def get_p_values(self):
        """
        Returns the coefficients of the p-values. 
        """
        return self.p

def run_regression(mode = "train", accuracy = 2, run_feature_engineering = False, include_dummies = False, plot_prediction = True):
    """
    Loads and preprocess training data. Fits the regression. If mode = "test", evaluates the model on testing data.

    Arguments:
    - mode: str, train or test to perform train only or train and test.
    - accuracy: int, length of the interval
    - run_feature_engineering: bool, forces to run the feature_engineering
    - include_dummies: bool, include_dummies in feature engineering
    - plot_prediction: bool, plots predictions.
    """
    if (os.path.exists("../data/taxitime_train_variables.csv") and not run_feature_engineering):
        taxitime_data = pd.read_csv("../data/taxitime_train_variables.csv")
        taxitime_data = taxitime_data.drop(["Unnamed: 0"], axis = 1)

    else: 
        print("Preprocessing training data ------ ")
        taxitime_data = ft.feature_engineering(include_dummies = include_dummies)

    reg = Regression()
    taxitime_data = taxitime_data.dropna(subset = ["windGust"])
    #print(np.sum(pd.isna(taxitime_data)))
    X_train, _, y_train, _ = reg.preprocess(taxitime_data, train_size = 0.99)

    print("Fitting model -----")
    reg = reg.fit(X_train, y_train)
    #print("Coefficients: ", reg.get_coef_())
    print("P-Values: ", reg.get_p_values())

    if(mode == "test"):

        if (os.path.exists("../data/taxitime_test_variables.csv") and not run_feature_engineering):
            test_data = pd.read_csv("../data/taxitime_test_variables.csv")
            test_data = test_data.drop(["Unnamed: 0"], axis = 1)

        else: 
            print("Preprocessing testing data ------ ")
            test_data = ft.feature_engineering(mode = "test", include_dummies = include_dummies)

        test_data = test_data.dropna(subset = ["windGust"])
        _, X_test, _, y_test = reg.preprocess(test_data, train_size = 0.01)
        print("Evaluating model -----")
        r2, mae, accuracy = reg.evaluate(X_test, y_test, interval = accuracy, plot_prediction=plot_prediction)
        print("Performances on test data: ")
        print("R2 Score: ", r2)
        print("MAE: ", mae)
        print("Accuracy: ", accuracy)
    #print("Prediction for ", reg.x_scaler.inverse_transform(X_test[0,:]).T, ": ",  reg.predict(X_test[0,:]), "\True value: ", reg.y_scaler.inverse_transform(y_test[0,:]).T)

#run_regression(mode = "test")


