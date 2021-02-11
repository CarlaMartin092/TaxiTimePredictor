import pandas as pd
import feature_engineering as ft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import numpy as np
from collections import defaultdict
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

class FRBS():
    def __init__(self, n_clusters, regression = False):
        self.n_clusters = n_clusters #ex: [2, 3, 6, 2] nb of clusters per y_split
        self.scaler = StandardScaler()

        self.centroids = defaultdict(lambda: defaultdict(float))
        self.stds = defaultdict(lambda: defaultdict(float))

    def preprocess(self, data, train_size):
        X = data.drop(["actual_taxi_out_sec"], axis = 1)
        y = data["actual_taxi_out_sec"]
        X_rescaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_rescaled, y, train_size = train_size)
        return X_train, X_test, y_train, y_test

    def build_clusters(self, variables, X, y):

        kmeans = KMeans(self.n_clusters)
        fitted_X = kmeans.fit(X)
        labels = kmeans.labels_.reshape(-1,1)
        self.labels = labels

        for label in np.unique(labels):
            index = np.where(labels == label)[0]
            
            for i in range(X.shape[1]):
                self.centroids[label][variables[i]] = X[:,i].ravel()[index].mean()
                self.stds[label][variables[i]] = X[:,i].ravel()[index].std()

            self.centroids[label]["actual_taxi_out_sec"] = y.ravel()[index].mean()
            self.stds[label]["actual_taxi_out_sec"] = y.ravel()[index].std()
        return

    #def predict(self, y):


def run_FRBS(n_clusters, run_feature_engineering = False, include_dummies = False, train_size = 0.7):
    if (os.path.exists("../data/taxitime_train_variables.csv") and not run_feature_engineering):
        taxitime_data = pd.read_csv("../data/taxitime_train_variables.csv")
        taxitime_data = taxitime_data.drop(["Unnamed: 0"], axis = 1)

    else: 
        taxitime_data = ft.feature_engineering(include_dummies = include_dummies)

    print(taxitime_data.head())
    model = FRBS(n_clusters)
    X_train, X_test, y_train, y_test = model.preprocess(taxitime_data, train_size = train_size)

    variables = taxitime_data.columns[taxitime_data.columns != "actual_taxi_out_sec"]
    print(variables)
    model.build_clusters(variables, X_train, y_train)

run_FRBS(4)
