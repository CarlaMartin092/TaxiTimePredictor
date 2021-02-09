import pandas as pd
import numpy as np 
import os

def feature_engineering():
    #print(os.path.dirname(__file__))
    path_to_weather_data = "../data/training_set_weather_data.csv"
    path_to_airport_data = "../data/training_set_airport_data.csv"
    path_to_geo_data = "../data/geographic_data.csv"

    taxitime_weather = pd.read_csv(path_to_weather_data)

    #Drop useless columns
    taxitime_weather = taxitime_weather.drop(["Flight Datetime", "Aircraft Model", "Aircraft Length", "Aircraft Span", "No. Engines", \
        "Airport Arrival/Departure", "Movement Type", "AOBT", "ATOT"], axis = 1)

    #print("Weather cleaning start ----- ")
    #Weather data cleaning:
    taxitime_weather.loc[:,['summary', 'precipIntensity', 'precipProbability',\
       'temperature', 'apparentTemperature', 'dewPoint', 'humidity',\
       'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover',\
       'uvIndex', 'visibility', 'precipType', 'precipAccumulation', 'ozone']] = taxitime_weather.loc[:,['summary', 'precipIntensity', 'precipProbability',\
       'temperature', 'apparentTemperature', 'dewPoint', 'humidity',\
       'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover',\
       'uvIndex', 'visibility', 'precipType', 'precipAccumulation', 'ozone']].fillna(method = "ffill")

    taxitime_weather["night"] = taxitime_weather.icon.apply(lambda summ: 1 if "night" in str(summ) else 0)
    taxitime_weather["y"] = taxitime_weather["actual_taxi_out_sec"]
    taxitime_weather = taxitime_weather.drop(['icon', 'actual_taxi_out_sec'], axis = 1)
    #print("Weather cleaning completed ----- ")
    print("Number of columns: ", len(taxitime_weather.columns))
    #pd.to_csv("../data/")
    return(taxitime_weather)

feature_engineering()



