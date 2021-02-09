import pandas as pd
import numpy as np 
from pathlib import Path
import os

def feature_engineering():
    #print(os.path.dirname(__file__))
    path_to_weather_data = "../data/training_set_weather_data.csv"
    path_to_airport_data = "../data/training_set_airport_data.csv"
    path_to_geo_data = "../data/geographic_data.csv"

    taxitime_weather = pd.read_csv(path_to_weather_data)

    print("Datetime conversion -----")
    taxitime_weather["Flight Datetime"] = pd.to_datetime(taxitime_weather["Flight Datetime"])
    taxitime_weather.ATOT = pd.to_datetime(taxitime_weather.ATOT)
    taxitime_weather.AOBT = pd.to_datetime(taxitime_weather.AOBT) 
    print("Datetime completed -----")

    print("Weather cleaning start ----- ")
    #Weather data cleaning:
    taxitime_weather.iloc[:,22:] = taxitime_weather.iloc[:,22:].fillna(method = "ffill")
    print("Weather cleaning completed ----- ")

    

    pd.write_csv("../data/")
    return(taxitime_weather)

feature_engineering()



