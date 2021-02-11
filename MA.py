#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class MA60():
    """
    Simple Moving Average of taxi times for the past 60 days to predict
    today's taxi time
    """

    def __init__(self):
        """
        Initializes dataframes necessary to perform computations
        """
        self.df = []
        self.df_format = []

    def format(self, date, df):
        """
        Compute number of flight per day
        Group by flight date to compute average taxi time

        Args:
        - date: datetime object - obtain daily taxi times
        - df: pandas dataframe to manipulate

        Returns:
        - df_format: pandas dataframe to calculate daily taxi time

        """
        self.df = df
        self.df['nb_flights_per_day'] = np.array(1)
        self.df_format= self.df.groupby([date]).sum()
        return self.df_format

    def get_mean(self, df_format):
        """
        Fill MA by nan so first 60 days of df are not empty
        Smooth for christmas, take average of christmas days as there is 5 to 6times less flights these dates

        Args:
        - df_format: pandas dataframe with dates as index

        Returns:
        - df_format: pandas dataframe with 60 MA values smoothed for christmas
        """
        self.df_format = df_format
        self.df_format["60 days MA"] = np.nan

        for i in range(60, self.df_format.shape[0]):

            self.df_format['60 days MA'][i] = self.df_format['actual_taxi_out_min'][i-60:i].mean()\
                                    /self.df_format['nb_flights_per_day'][i]

        #smoothing for christmas day
            xmas_taxi_time = 0
            xmas_nb_flights =0
            if (self.df_format.index[i].month == 12) and (self.df_format.index[i].day== 25):
                xmas_taxi_time += self.df_format['actual_taxi_out_min'][i]
                xmas_nb_flights += self.df_format['nb_flights_per_day'][i]
                self.df_format['60 days MA'][i] = xmas_taxi_time /xmas_nb_flights

        return self.df_format


    def merge(self, df_format, date):
        """
        Merging the grouped by df with initial df so that every flight has a daily average

        Args:
        - date: datetime object, to perform merge operation on
        - df_format: pandas df to perform merge operation with, on inital df

        Returns:

        """
        self.df = pd.merge(self.df, df_format, on = date)
        self.df = self.df[['actual_taxi_out_min_x', 'Flight Date', '60 days MA']].rename(columns={"actual_taxi_out_min_x": "actual_taxi_out_min"})
        return self.df


    def get_acc_metrics(self, df, split, interval):
        """
        Calculate MAE, RMSE, confidence interval of predictions under 2minutes of true value
        on test set of size (1-split)%

        Args:
        - df: pandas df - to perform operations on
        - split: int - to split dataset
        - interval: int - interval of minutes

        Returns:
        - MAE, RMSE, confidence interval
        """
        train_size= int(split*len(self.df))
        train_set = df[:train_size]
        test_set = df[train_size:]

        mae = mean_absolute_error(test_set['actual_taxi_out_min'], test_set['60 days MA'])
        rmse =  np.sqrt(mean_squared_error(test_set['actual_taxi_out_min'], test_set['60 days MA']))
        acc_int = np.abs(test_set['actual_taxi_out_min'] - test_set['60 days MA'])
        acc_int = acc_int <= interval
        acc_int = np.mean(acc_int)
        return mae, rmse, acc_int


def run():
    trainset_weather_data = pd.read_csv('../data/taxitime_training_variables.csv')
    trainset_weather_data['actual_taxi_out_min'] = trainset_weather_data['actual_taxi_out_sec']/60
    trainset_weather_data['Flight Datetime'] = pd.to_datetime(trainset_weather_data['Flight Datetime'], format = '%m/%d/%Y %H:%M')
    trainset_weather_data['Flight Date'] = [trainset_weather_data['Flight Datetime'][i].date() for i in range(0, len(trainset_weather_data['Flight Datetime']))]
    df = trainset_weather_data[['actual_taxi_out_min', 'Year', 'Month','Flight Date']]
    ma = MA60()
    df_format = ma.format('Flight Date',df)
    get_mean = ma.get_mean(df_format)
    df = ma.merge(get_mean, 'Flight Date')
    mae, rmse, acc_int = ma.get_acc_metrics(df, 0.7, 2)
    return df, mae, rmse, acc_int
