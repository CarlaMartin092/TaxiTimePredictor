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
    def __init__(self, quantiles, n_clusters):
        self.quantiles = quantiles #ex: [0, 0.2, 0.5, 0.8, 1.0]
        self.n_clusters = n_clusters #ex: [2, 3, 6, 2] nb of clusters per y_split
        self.scaler = StandardScaler()

        self.centroids = defaultdict(lambda: defaultdict(float))
        self.stds = defaultdict(lambda: defaultdict(float))

    def preprocess(self, data, train_size):
        X = data.drop(["y"], axis = 1)
        y = data["y"]
        X_rescaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_rescaled, y, train_size = train_size)
        return X_train, X_test, y_train, y_test

    def build_clusters(self, X, y):
        
        data_labels = ['']*len(X)
        #print(len(data_labels))
        for i in range(len(self.n_clusters)):
            inf_bound = np.percentile(y, self.quantiles[i])
            sup_bound = np.percentile(y, self.quantiles[i+1])
            #print(sup_bound)
            selection = (y>inf_bound) & (y<=sup_bound)
            index_selection = selection[selection == True].reset_index().index.values
            #print(len(index_selection))
            #print("Number of observations selected: ", np.sum(selection))
            X_selected = X[selection]
            #print(X_selected.shape)

            kmeans = KMeans(self.n_clusters[i])
            fitted_x = kmeans.fit(X_selected)
            labels = kmeans.labels_.reshape(-1,1)
            labels = ['q{}_'.format(i+1) + str(l[0]) for l in labels]
            #print(labels[0:5], len(labels))
            #print(np.max(index_selection))

            for i in range(len(index_selection)):
                data_labels[index_selection[i]] = labels[i]
            
            
        
        return

    def add_label(self, data):
        """
        Adds the cluster label to a row of data, based on the quantile values and the number of clusters.
        """
        return

def run_FRBS(quantiles, n_clusters, run_feature_engineering = False, include_dummies = False, train_size = 0.7):
    if (os.path.exists("../data/taxitime_variables.csv") and not run_feature_engineering):
        taxitime_data = pd.read_csv("../data/taxitime_variables.csv")
        taxitime_data = taxitime_data.drop(["Unnamed: 0"], axis = 1)

    else: 
        taxitime_data = ft.feature_engineering(include_dummies = include_dummies)

    model = FRBS(quantiles, n_clusters)
    X_train, X_test, y_train, y_test = model.preprocess(taxitime_data, train_size = train_size)
    model.build_clusters(X_train, y_train)

run_FRBS([0.0, 0.5, 1.0], [2, 2])
