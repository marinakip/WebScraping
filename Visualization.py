from turtle import pd
from geopy import Nominatim
import pandas as pd
from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium


# df = pd.read_csv("addresses_all.csv", header = None)
# print(df)

string = "Barkhamsted, CT"
df = pd.DataFrame([string], columns=['string_values'])
print(df)
locator = Nominatim(user_agent="myGeocoder")
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
#df['location'] = locator.geocode("Champ de Mars, Paris, France")
df['location'] = df.apply(geocode)
# 2- - create location column
#df['location'] = df.apply(geocode)
# 3 - create longitude, laatitude and altitude from location column (returns tuple)
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# 4 - split point column into latitude, longitude and altitude columns
df[['latitude', 'longitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

map1 = folium.Map(
    location=[59.338315,18.089960],
    tiles='cartodbpositron',
    zoom_start=12,
)
df.apply(lambda row:folium.CircleMarker(location = [row["latitude"], row["longitude"]]).add_to(map1), axis=1)
map1




# locator = Nominatim(user_agent="myGeocoder")
# location = locator.geocode("Champ de Mars, Paris, France")
# print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))



