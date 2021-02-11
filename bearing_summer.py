# adding some variables
def is_summer(x):
    if x in [5, 6, 7, 8, 9, 10]:
        return 1
    else:
        return 0

df_original['Summer'] = df_original['Month'].map(lambda x : is_summer(x))

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

df_original['wind_direction'] = df_original['windBearing'].map(lambda x : wind_direction(x))
res = pd.get_dummies(df_original['wind_direction'])

df_original = pd.merge(df_original, res, left_index=True, right_index=True)
