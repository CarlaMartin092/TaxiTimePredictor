import pandas as pd
import numpy as np 
import os

def compute_N(df):
    """
    Computes the number of aircrafts taxiing at the same time as another 
    """

    df = df.sort_values('AOBT')
    AOBT = list(df["AOBT"])
    ATOT = list(df["ATOT"])
    n_flights = len(AOBT)
    N = {}
    index = list(df.index)
    for i in range(n_flights):
        if(i == 0): N[index[i]] = 0
        else:
            k = 1
            while(i - k >= 0 and AOBT[i] >= AOBT[i - k] and ATOT[i - k] >= AOBT[i]):
                k+=1
            N[index[i]] = k - 1
    res = [v for (k, v) in sorted(N.items())]
    return res

def compute_Q(df):
    df = df.sort_values('ATOT')
    AOBT = list(df["AOBT"])
    ATOT = list(df["ATOT"])
    n_flights = len(AOBT)
    Q = {}
    index = list(df.index)
    for i in range(n_flights):
        if(i == 0): Q[index[i]] = 0
        else:
            k = 1
            while(i - k >= 0 and AOBT[i] <= ATOT[i - k] and ATOT[i] >= ATOT[i - k]):
                k+=1
            Q[index[i]] = k - 1
    res = [v for (k, v) in sorted(Q.items())]
    return res

def wind_direction(x):
    if 315.0 < x <= 360.0:
        return 'North'
    if 0.0 <= x <= 45.0:
        return 'North'
    if 45.0 < x <= 135.0:
        return 'East'
    if 135.0 < x <= 225.0:
        return 'South'
    if 225.0 < x <= 315.0:
        return 'West'

def feature_engineering(mode = "train", include_dummies = False):
    #print(os.path.dirname(__file__))
    if(mode == "train"):
        path_to_weather_data = "../data/training_weather_data.csv"
        path_to_airport_data = "../data/training_set_airport_data.csv"
        path_to_geo_data = "../data/geographic_data.csv"

        weather = pd.read_csv(path_to_weather_data)
        airport = pd.read_csv(path_to_airport_data)
    else:
        path_to_weather_data = "../data/test_set_weather_data.xlsx"
        path_to_airport_data = "../data/test_set_airport_data.xlsx"
        path_to_geo_data = "../data/geographic_data.csv"

        weather = pd.read_excel(path_to_weather_data)
        airport = pd.read_excel(path_to_airport_data)
    
    #print(len(airport))
   # geo_data = pd.read_csv(path_to_geo_data)
    extended = pd.read_csv("../data/training_set_weather_data.csv")
    extended = extended[['Flight Datetime', 'Aircraft Model', 'Aircraft Length', 'Aircraft Span', 'AOBT', 'ATOT', 'Distance_proxy_m']]
   
    extended["AC_size"] = extended['Aircraft Length'] * extended["Aircraft Span"]
    extended = extended.drop(['Aircraft Length', 'Aircraft Span'], axis = 1)

    extended["Flight Datetime"] = pd.to_datetime(extended["Flight Datetime"])
    extended["AOBT"] = pd.to_datetime(extended["AOBT"])
    extended["ATOT"] = pd.to_datetime(extended["ATOT"])
    #print(extended.info())
    
    #print(len(weather))
    weather = weather.drop_duplicates(subset = ["time_hourly"])
    weather["time_hourly"] = pd.to_datetime(weather["time_hourly"])

    weather = weather.drop(['apparentTemperature', 'dewPoint', 'pressure', 'cloudCover', 'precipAccumulation', 'ozone', 'uvIndex', 'precipProbability', 'icon', 'summary'], axis = 1)

    airport["Flight Datetime"] = pd.to_datetime(airport["Flight Datetime"])
    airport["AOBT"] = pd.to_datetime(airport["AOBT"])
    airport["ATOT"] = pd.to_datetime(airport["ATOT"])
    
    airport["Date"] = airport["Flight Datetime"].dt.date
    airport["Hour"] = airport["Flight Datetime"].dt.hour
    airport["time_hourly"] = pd.to_datetime(airport["Date"].map(str) + ' ' + airport["Hour"].map(str) + ':00')
    airport = airport.drop(["Hour", "Date"], axis = 1)

    distances = airport.merge(extended, how = 'left', on = ['Flight Datetime','Aircraft Model', 'AOBT'])
    distances = distances.groupby(['Stand','Runway'])

    distance_mean = distances['Distance_proxy_m'].mean().reset_index()
    #print(distance_mean)

    distance_std = distances['Distance_proxy_m'].std().reset_index()
    distance_std['Distance_std'] =  distance_std['Distance_proxy_m']
    distance_std = distance_std.drop('Distance_proxy_m', axis = 1)

    distances = distance_mean.merge(distance_std, on = ['Stand','Runway'])

    #airport = airport.merge(extended[['Aircraft Model', 'AC_size']], how = 'left', on = ["Aircraft Model"])
    airport = airport.merge(distances, how = "left", on = ['Stand', 'Runway'])
    airport = airport.drop(['Stand', 'Runway'], axis = 1)

    taxitime_weather = airport.merge(weather, how = "left", on = 'time_hourly')

    taxitime_weather["morning_mode"] = taxitime_weather['time_hourly'].dt.hour.apply(lambda h: 1 if (h >= 5) and (h <= 18) else 0)
    taxitime_weather["summer"] = taxitime_weather["time_hourly"].dt.month.apply(lambda x: 1 if x in [5, 6, 7, 8, 9, 10] else 0)
    taxitime_weather = taxitime_weather.drop(["time_hourly", "Flight Datetime"], axis = 1)

    taxitime_weather["Log_distance_m"] = np.log(taxitime_weather["Distance_proxy_m"])
    taxitime_weather["actual_taxi_out_sec"] = (taxitime_weather["ATOT"] - taxitime_weather["AOBT"]) / np.timedelta64(1, 's')
    taxitime_weather["N"] = compute_N(taxitime_weather)
    taxitime_weather["Q"] = compute_Q(taxitime_weather)
    taxitime_weather["windBearing"] = taxitime_weather["windBearing"].apply(lambda x: wind_direction(x))
    #print(taxitime_weather["N"])
    #print(taxitime_weather["Q"])
    taxitime_weather = taxitime_weather.drop(["Aircraft Model", "ATOT", "AOBT"], axis = 1)

    taxitime_weather.loc[:,["precipIntensity", "temperature", "windSpeed", "visibility", "precipType", "windGust", "humidity", "windBearing"]] =  taxitime_weather.loc[:,["precipIntensity", "temperature", "windSpeed", "visibility", "precipType", "windGust", "humidity", "windBearing"]].fillna(method = "ffill")
    
    #print("NB missing values taxitime_weather: ", np.sum(pd.isna(taxitime_weather)))

    if(include_dummies):

        taxitime_weather["precipType"] = taxitime_weather["precipType"].apply(lambda w: w if w == "rain" else "None")
        taxitime_weather = pd.get_dummies(taxitime_weather, columns = ["precipType", "windBearing"])
        taxitime_weather = taxitime_weather.drop(['precipType_None', "windBearing_West"], axis = 1)
        taxitime_weather['precipType_rain'] = taxitime_weather["precipIntensity"] * taxitime_weather['precipType_rain']
        taxitime_weather['precipType_rain'] = taxitime_weather['precipType_rain'].fillna(0)
        taxitime_weather = taxitime_weather.drop(["precipIntensity"], axis = 1)

    else:
        taxitime_weather = taxitime_weather.drop(["precipType", "precipIntensity"], axis = 1)

    print("NB missing values taxitime_weather: ", np.sum(pd.isna(taxitime_weather)))
    print(taxitime_weather.columns)
    path_file = "../data/taxitime_{}_variables.csv".format(mode)
    taxitime_weather.to_csv(path_file)

    return(taxitime_weather)
    
#feature_engineering(mode = "train", include_dummies=True)
#feature_engineering(mode = "test", include_dummies=True)



