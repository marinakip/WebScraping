from folium.plugins import FastMarkerCluster
import pandas as pd
import folium

df = pd.read_csv("addresses_geocoded.csv", index_col=0)
#print(df.head(5))
df = df.dropna()
df = df.reset_index(drop=True)
#print(df)

folium_map = folium.Map(location=[37.983810, 23.727539],
                        zoom_start=2,
                        tiles='CartoDB dark_matter')
#folium_map.save('map.html')

FastMarkerCluster(data=list(zip(df['Latitude'].values, df['Longitude'].values))).add_to(folium_map)
folium.LayerControl().add_to(folium_map)
folium_map

folium_map.save('map.html')


