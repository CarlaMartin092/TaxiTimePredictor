import pandas as pd
import numpy as np 
import os

def feature_engineering(include_dummies = False):
    #print(os.path.dirname(__file__))
    path_to_weather_data = "../data/training_set_weather_data.csv"
    path_to_airport_data = "../data/training_set_airport_data.csv"
    path_to_geo_data = "../data/geographic_data.csv"

    taxitime_weather = pd.read_csv(path_to_weather_data)

    taxitime_weather["Hour"] = pd.to_datetime(taxitime_weather["Flight Datetime"]).dt.hour
    taxitime_weather["morning_mode"] = taxitime_weather['Hour'].apply(lambda h: 1 if (h >= 5) and (h <= 18) else 0)

    print(np.sum(pd.isna(taxitime_weather['Distance_proxy_m'])))
    #taxitime_weather['Distance_proxy_m'] = taxitime_weather.groupby(['Aircraft Model','Hour', 'Weekday', 'Month', 'Year'])['Distance_proxy_m'].transform(lambda x: x.fillna(x.mode()) if not x.mode().empty else taxitime_weather['Distance_proxy_m'].mean())
    taxitime_weather = taxitime_weather.dropna(axis = 0, subset = ["Distance_proxy_m"])
    print(np.sum(pd.isna(taxitime_weather['Distance_proxy_m'])))
    taxitime_weather['Log_distance_m'] = np.log(taxitime_weather['Distance_proxy_m'])
    #Drop useless columns
    taxitime_weather = taxitime_weather.drop(["Flight Number", "Flight Datetime", "Aircraft Model", "Aircraft Length", "Aircraft Span", "No. Engines", \
        "Airport Arrival/Departure", "Distance_proxy_m", "Movement Type", "AOBT", "ATOT", "time_hourly", 'apparentTemperature', 'dewPoint',\
        'humidity', 'pressure', 'windGust', 'windBearing', 'cloudCover', 'precipProbability', 'ozone', 'uvIndex', 'Hour'], axis = 1)

    #print("Weather cleaning start ----- ")
    #Weather data cleaning:
    taxitime_weather.loc[:,['summary', 'precipIntensity',\
       'temperature', 'windSpeed', 'visibility', 'precipType', 'precipAccumulation']] = taxitime_weather.loc[:,['summary', 'precipIntensity',\
       'temperature', 'windSpeed', 'visibility', 'precipType', 'precipAccumulation']].fillna(method = "ffill")

    #print(sum(pd.isna(taxitime_weather.summary)))
    #print(sum(pd.isna(taxitime_weather.precipType)))

    taxitime_weather["night"] = taxitime_weather.icon.apply(lambda summ: 1 if "night" in str(summ) else 0)
    taxitime_weather["y"] = taxitime_weather["actual_taxi_out_sec"]
    taxitime_weather = taxitime_weather.drop(['icon', 'actual_taxi_out_sec'], axis = 1)
    #print("Weather cleaning completed ----- ")

    if(include_dummies):

        taxitime_weather = pd.get_dummies(taxitime_weather, columns = ["Year", "Month", "Weekday", "summary", "precipType"])
        taxitime_weather = taxitime_weather.drop(['Year_2015', 'Month_1', 'Weekday_0', 'precipType_None', 'summary_Windy and Partly Cloudy'], axis = 1)
    
    else:
        taxitime_weather = taxitime_weather.drop(["Year", "Month", "Weekday", "summary", "precipType"], axis = 1)

    #taxitime_weather = taxitime_weather.drop(['summary_0', 'Month_1', 'Weekday_0'], axis = 1)

    print(taxitime_weather.columns)
    print("Number of columns: ", len(taxitime_weather.columns))
    taxitime_weather.to_csv("../data/taxitime_variables.csv")
    return(taxitime_weather)

#feature_engineering()



