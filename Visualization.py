from turtle import pd
from geopy import Nominatim
import pandas as pd
from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import time
import datetime
import random


df = pd.read_csv("addresses_cleaned.csv", sep='\"', header = None)
#print(df)

geocoding_results = []
counter = 0

for row in range(len(df)):
    counter += 1
    if counter % 30 == 0:
        print("SLEEP")
        secs = random.randint(20, 60)
        print("SLEEP " + str(secs))
        time.sleep(secs)
    print("GEOCODING LOCATION:  " + str(counter))
    address = df.iloc[row].values
    #print(address)
    #print(address.to_string())
    locator = Nominatim(user_agent="myGeocoder")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=2)
    try:
        location = address.apply(geocode)
        #location = locator.geocode(address)
        latitude = location.latitude
        longitude = location.longitude
    except AttributeError:
        location = "NONE"
        latitude = "NONE"
        longitude = "NONE"
    #print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
    dictionary = {
        'Address': address,
        'Latitude': latitude,
        'Longitude': longitude
    }
    geocoding_results.append(dictionary)

df_addresses = pd.DataFrame(geocoding_results)
df_addresses.to_csv('addresses_geocoded.csv')
print("CSV GEOCODING ADDRESSES TEMP CREATED")


#
#
# string =  #"Barkhamsted, CT"
# df = pd.DataFrame([string], columns=['string_values'])
# #print(df)
# locator = Nominatim(user_agent="myGeocoder")
# geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
# #df['location'] = locator.geocode("Champ de Mars, Paris, France")
# df['location'] = df.apply(geocode)
# # 2- - create location column
# #df['location'] = df.apply(geocode)
# # 3 - create longitude, laatitude and altitude from location column (returns tuple)
# df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# # 4 - split point column into latitude, longitude and altitude columns
# df[['latitude', 'longitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)
#
# map1 = folium.Map(
#     location=[59.338315,18.089960],
#     tiles='cartodbpositron',
#     zoom_start=12,
# )
# df.apply(lambda row:folium.CircleMarker(location = [row["latitude"], row["longitude"]]).add_to(map1), axis=1)
# map1
#



