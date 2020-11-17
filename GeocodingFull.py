import pandas as pd
from geopy import Nominatim
import time
import random

#df = pd.read_csv("scraping_results_weighted.csv", header = 0)
df = pd.read_csv("scraping_results_cleaned_diabetes_combinedAll_weighted.csv", header = 0)
#print(df.head(10))

data = df['Location']
#print(data.head(10))

geocoding_results = []
counter = 0
latitude_list = []
longitude_list = []

for row in range(len(data)):
    counter += 1
    if counter % 30 == 0:
        print("SLEEP")
        secs = random.randint(20, 60)
        print("SLEEP " + str(secs))
        time.sleep(secs)
    print("GEOCODING LOCATION:  " + str(counter))
    address = data.iloc[row]
    print(address)
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
    # dictionary = {
    #     'Address': address,
    #     'Latitude': latitude,
    #     'Longitude': longitude
    # }
    latitude_list.append(latitude)
    longitude_list.append(longitude)
    #geocoding_results.append(dictionary)
    # if counter%20 == 0:
    #     df_addresses = pd.DataFrame(geocoding_results)
    #     df_addresses.to_csv('addresses_geocoded_temp_weighted.csv')
    #     print("CSV GEOCODING ADDRESSES TEMP CREATED")

df['Longitude'] = longitude_list
df['Latitude'] = latitude_list

#df_addresses = pd.DataFrame(geocoding_results)
#df.to_csv('addresses_geocoded_weighted_full.csv', index=False, header=True)
df.to_csv('scraping_results_cleaned_diabetes_combinedAll_weighted_geocoded.csv', index=False, header=True)
print("CSV GEOCODING ADDRESSES FINAL CREATED")







