import pandas as pd
from geopy import Nominatim
import time
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
    #geocode = RateLimiter(locator.geocode, min_delay_seconds=2)
    try:
        #location = address.apply(geocode)
        location = locator.geocode(address, timeout= 30)
        latitude = location.latitude
        longitude = location.longitude
    except AttributeError:
        #location = "NONE"
        latitude = None
        longitude = None
    #print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
    dictionary = {
        'Address': address,
        'Latitude': latitude,
        'Longitude': longitude
    }
    geocoding_results.append(dictionary)
    if counter%20 == 0:
        df_addresses = pd.DataFrame(geocoding_results)
        df_addresses.to_csv('addresses_geocoded_temp.csv')
        print("CSV GEOCODING ADDRESSES TEMP CREATED")

df_addresses = pd.DataFrame(geocoding_results)
df_addresses.to_csv('addresses_geocoded.csv')
print("CSV GEOCODING ADDRESSES FINAL CREATED")


