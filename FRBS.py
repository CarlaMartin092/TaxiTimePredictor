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
from sklearn.metrics  import r2_score, mean_absolute_error
from metrics import accuracy
import os
import math


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

        outliers = y.quantile(0.95)
        #index_not_outliers = np.where(y[y <= outliers])
        y_no_outliers = y[y <= outliers]
        X = X[y <= outliers]
        y = y_no_outliers

        X_stack_y = np.concatenate((X, np.array(y)[:,None]), axis = 1)

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(X_stack_y)
        labels = kmeans.labels_.reshape(-1,1)
        self.labels = labels

        for label in np.unique(labels):
            index = np.where(labels == label)[0]
            
            for i in range(X_stack_y.shape[1]):
                self.centroids[label][variables[i]] = X_stack_y[:,i].ravel()[index].mean()
                self.stds[label][variables[i]] = X_stack_y[:,i].ravel()[index].std()

            #self.centroids[label]["actual_taxi_out_sec"] = y.ravel()[index].mean()
            #self.stds[label]["actual_taxi_out_sec"] = y.ravel()[index].std()
        return

    def membership(self, cluster, variable, x):
        centroid = self.centroids[cluster][variable]
        #print(centroid)
        std = self.stds[cluster][variable]
        result = x - centroid
        result = -0.5*(result/(std + 1e-5))**2
        result = np.exp(result)
        return result 

    def total_membership(self, cluster,x):

        result = 1
        variables = list(self.centroids[cluster].keys())
        for v in range(len(variables)):
            if variables[v] == "actual_taxi_out_sec" or variables[v] == "label": continue
            result *= self.membership(cluster, variables[v], x[v])
        if(np.sum(result) == 0): result+=1e-6
        return result

    def output_membership(self, cluster, y):
        centroid_y = self.centroids[cluster]["actual_taxi_out_sec"]
        std_y = self.stds[cluster]["actual_taxi_out_sec"]
        den = 1 + ((y - centroid_y)/(std_y + 1e-5))**2
        return (1/den)

    def compute_integral(self, cluster):
        """
        Computes the discretized version of the integral of the output membership for a given cluster i.
        """
        centroid_y = self.centroids[cluster]["actual_taxi_out_sec"]
        std_y = self.stds[cluster]["actual_taxi_out_sec"]
        integral = std_y * (math.pi/2 - math.atan(-centroid_y/(std_y + 1e-6)))
        return(integral)

    def y_crisp(self, clusters, centroids_y, integrals_y, x):
        """
        Computes the y_crisp value described in the paper "On the Utilisation of Fuzzy Rule-Based Systems
    for Taxi Time Estimations at Airports", Chen et al.

        Arguments:
        - clusters: list of strings, list of clusters names
        - y_set: np.array, training set of y values
        - x: np.array, indenpendent variables associated to y
        """
        integrals = integrals_y
        total_membership_values = np.array([self.total_membership(c, x) for c in clusters])
        sum_integral_clusters = np.sum(total_membership_values * integrals)
        weighted_sum = np.sum(centroids_y * total_membership_values * integrals)
        if(np.isnan(weighted_sum/sum_integral_clusters)): return
        return(weighted_sum/sum_integral_clusters)

    def predict(self, X):

        clusters = list(self.centroids.keys())
        centroids_y = np.array([self.centroids[c]["actual_taxi_out_sec"] for c in clusters]) 
        integrals_y = np.array([self.compute_integral(c) for c in clusters])
        y_pred = self.y_crisp(clusters, centroids_y, integrals_y, X)
        return(y_pred)

    def evaluate(self, X_test, y_test):

        predictions = [self.predict(X_test[i,:]) for i in range(len(X_test))]
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        acc = accuracy(y_test, predictions)
        return(r2, mae, acc)

    def show_rules(self, data):
        cols = data.columns
        #cols = cols[cols != "actual_taxi_out_sec"]
        percentiles = data[cols].quantile([0.05,0.95])

        for cluster in self.centroids:
            f, axs = plt.subplots(nrows = 1, ncols = len(cols), figsize = (200,10))
            f.suptitle('Rule: {}'.format(cluster), fontsize=30)
            for i, variable in enumerate(cols):
                minimum = percentiles[variable].iloc[0]
                maximum = percentiles[variable].iloc[1]
                x = list(np.linspace(minimum, maximum, 1000))
                y = self.membership(cluster, variable, x)
                axs[i].scatter(x, y)
                axs[i].set_xlabel(variable, fontsize=30)
                axs[i].set_ylabel("Membership degree, cluster{}".format(cluster), fontsize=20)
            plt.savefig("../data/Rule_cluster{}.png".format(cluster))
        

def run_FRBS(n_clusters, mode = "train", run_feature_engineering = False, include_dummies = False, train_size = 0.99):

    if (os.path.exists("../data/taxitime_train_variables.csv") and not run_feature_engineering):
        taxitime_data = pd.read_csv("../data/taxitime_train_variables.csv")
        taxitime_data = taxitime_data.drop(["Unnamed: 0"], axis = 1)

    else: 
        taxitime_data = ft.feature_engineering(mode = "train", include_dummies = include_dummies)

    #print(taxitime_data.head())
    taxitime_data = taxitime_data[["Log_distance_m","N","Q","windSpeed","visibility","humidity","windGust","Distance_std","actual_taxi_out_sec"]]
    print(taxitime_data.head())
    model = FRBS(n_clusters)
    X_train, _, y_train, _ = model.preprocess(taxitime_data, train_size = train_size)

    variables = taxitime_data.columns
    model.build_clusters(variables, X_train, y_train)
    variables = taxitime_data.columns[taxitime_data.columns != "actual_taxi_out_sec"]
    #print(variables)
    model.show_rules(taxitime_data)

    if(mode == "test"):
        if (os.path.exists("../data/taxitime_test_variables.csv") and not run_feature_engineering):
            test_data = pd.read_csv("../data/taxitime_test_variables.csv")
            test_data = test_data.drop(["Unnamed: 0"], axis = 1)

        else: 
            test_data = ft.feature_engineering(mode = "test", include_dummies = include_dummies)

        test_data = test_data[["Log_distance_m","N","Q","windSpeed","visibility","humidity","windGust","Distance_std","actual_taxi_out_sec"]]
        test_data = test_data.dropna(subset = ["windGust"])
        _, X_test, _, y_test = model.preprocess(test_data, train_size = 0.01)
        r2, mae, accuracy = model.evaluate(X_test, y_test)
        print("Performances on test data: ")
        print("R2: ", r2)
        print("MAE: ", mae)
        print("Accuracy: ", accuracy)
    
run_FRBS(11, mode = "test")
