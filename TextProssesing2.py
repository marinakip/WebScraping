import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
#from plotly.offline import iplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from geopy import Nominatim
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import geopandas
import matplotlib.pyplot as plt
import geoplot
import mapclassify
import geoplot.crs as gcrs
import mapclassify
import geojson

if not os.path.exists("map_images"):
    os.mkdir("map_images")

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
english = set(nltk.corpus.words.words())


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


def clean_text(text):
    words = re.sub(r'\\n', ' ', text)
    cleaned = re.sub(r'\W+', ' ', words).lower()
    tokenized = word_tokenize(cleaned)
    filtered = [word for word in tokenized if word not in stop_words and word in english]
    normalized = " ".join([lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in filtered])
    return normalized

def normalize(df):
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result

def weight(df):
    df['Weight'] = (df['Followers'] * 0.2) + (df['Following'] * 0.1) + (df['Stars'] * 0.3) + (
                df['Contributions'] * 0.4)
    return df['Weight']

def process_query(query):
    data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
    cleaned_text = [clean_text(text) for text in data['Description']]

    vectorizer = TfidfVectorizer(norm = None)
    tfidf_scores = vectorizer.fit_transform(cleaned_text)

    query_vec = vectorizer.transform([query])

    results = cosine_similarity(tfidf_scores, query_vec)

    results_position = np.where(results)

    sorted = np.argsort(results[results_position])[::-1]  # descending order

    results_sorted = tuple(np.array(results_position)[:, sorted])

    geocoding_results = []

    print("GEOCODING RESULTS STARTED ")

    for i in results_sorted[0]:
        similarity = results[i]
        similarity = str(similarity).replace('[', '').replace(']', '')
        address = data.iloc[i, 0]
        locator = Nominatim(user_agent = "myGeocoder")
        followers = data.iloc[i, 1]
        following = data.iloc[i, 2]
        stars = data.iloc[i, 3]
        contributions = data.iloc[i, 4]
        url_profile = data.iloc[i, 6]
        # info

        try:
            location = locator.geocode(address, timeout = 30)
            latitude = location.latitude
            longitude = location.longitude
            location = locator.reverse([latitude, longitude])

            display_name = location.raw['display_name']
            country = location.raw['address']['country']
            country_code = location.raw['address']['country_code']

        except AttributeError:

            latitude = None
            longitude = None

        except KeyError:
            country = None
            country_code = None
        except NameError:

            country = None
            country_code = None

        finally:

            dictionary = {
                'Address': address,
                'Latitude': latitude,
                'Longitude': longitude,
                'Country': country,
                'Country_Code': country_code,
                'Followers': followers,
                'Following': following,
                'Stars': stars,
                'Contributions': contributions,
                # 'Description': description
                'Url_profile': url_profile,
                'Similarity': similarity
                # Info : info

            }
            geocoding_results.append(dictionary)

    df = pd.DataFrame(geocoding_results)

    print("NORMALIZATION OF RESULTS")

    # print("Followers Normalized")
    norm_followers = normalize(df.Followers)
    # print(norm_followers.head())
    df.Followers = norm_followers

    # print("Following Normalized")
    norm_following = normalize(df.Following)
    # print(norm_following.head())
    df.Following = norm_following

    # print("Stars Normalized")
    norm_stars = normalize(df.Stars)
    # print(norm_stars.head())
    df.Stars = norm_stars

    # print("Contributions Normalized")
    norm_contributions = normalize(df.Contributions)
    # print(norm_contributions.head())
    df.Contributions = norm_contributions

    print("WEIGHT")
    df.Weight = weight(df)

    # print(df.head(10))

    print("PLOTTING CHOROPLETH")

    path_to_file = 'custom.geo.json'
    with open(path_to_file) as f:
        geo = geojson.load(f)

    fig = px.choropleth(data_frame = df,
                        geojson = geo,
                        locations = 'Country',
                        locationmode = 'country names',
                        color = 'Followers',
                        color_continuous_scale = 'Viridis',
                        range_color = (0, 1))

    #return df
    return fig, df



# #-------------------------------------------------------------------------------------
# data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
# cleaned_text = [clean_text(text) for text in data['Description']]
# #print(cleaned_text)
#
#
# vectorizer = TfidfVectorizer(norm=None)
# tfidf_scores = vectorizer.fit_transform(cleaned_text)
# #print(tfidf_scores)
#
# query = "glucose meter"
# query_vec = vectorizer.transform([query])
#
# results = cosine_similarity(tfidf_scores, query_vec)
# #print(results)
#
#
# results_position = np.where(results)
#
# #print(results_position)
# sorted = np.argsort(results[results_position])[::-1]  # descending order
# #print("SORTED")
# #print(sorted)
# #print(" RESULT SORTED")
# results_sorted = tuple(np.array(results_position)[:, sorted])
# #print(results_sorted)
# #print("COSINE SIMILARITY RESULTS SORTED")
# cosine_similarity_sorted = results[results_sorted[0]]
# #print(cosine_similarity_sorted)
#
# # print("ELEMENTS VALUES")
# # print(results[169])
# # print(results[149])
# # print(results[279])
#
#
# #print(len(results))
# #print(len(data['Description']))
#
#
# # results_nonzero = np.nonzero(results)[0]
# # print(results_nonzero)
#
#
# geocoding_results = []
#
# #print("Length Result Addresses: {}".format(len(results_nonzero)))
#
# # for i in results_nonzero:
# for i in results_sorted[0]:
#     similarity = results[i]
#     similarity = str(similarity).replace('[', '').replace(']', '')
#     #print(similarity)
#     address = data.iloc[i, 0]
#     locator = Nominatim(user_agent="myGeocoder")
#     followers = data.iloc[i, 1]
#     #print(address)
#     #print(followers)
#     following = data.iloc[i, 2]
#     stars = data.iloc[i, 3]
#     contributions = data.iloc[i, 4]
#     # description = data.iloc[i, 5]
#     url_profile = data.iloc[i, 6]
#     # info
#
#     try:
#         location = locator.geocode(address, timeout = 30)
#         latitude = location.latitude
#         longitude = location.longitude
#         location = locator.reverse([latitude, longitude])
#         #print(location.raw)
#         display_name = location.raw['display_name']
#         country = location.raw['address']['country']
#         country_code = location.raw['address']['country_code']
#         #print(location.raw)
#     except AttributeError:
#         # location = "NONE"
#         latitude = None
#         longitude = None
#
#     except KeyError:
#         country = None
#         country_code = None
#     except NameError:
#         #print("MPIKE")
#         country = None
#         country_code = None
#
#     finally:
#         # print("Display Name = {}".format(display_name))
#         # print("Latitude = {}, Longitude = {}".format(latitude, longitude))
#         # print("Country = {}, Country Code = {}".format(country, country_code))
#
#         dictionary = {
#             'Address': address,
#             'Latitude': latitude,
#             'Longitude': longitude,
#             'Country': country,
#             'Country_Code': country_code,
#             'Followers'	: followers,
#             'Following'	: following,
#             'Stars'	: stars,
#             'Contributions'	: contributions,
#             #'Description': description
#             'Url_profile'	: url_profile,
#             'Similarity' : similarity
#             #Info : info
#
#         }
#         geocoding_results.append(dictionary)
#
# df = pd.DataFrame(geocoding_results)
#
# print("Followers Normalized")
# norm_followers = normalize(df.Followers)
# print(norm_followers.head())
# df.Followers = norm_followers
#
# print("Following Normalized")
# norm_following = normalize(df.Following)
# print(norm_following.head())
# df.Following = norm_following
#
# print("Stars Normalized")
# norm_stars = normalize(df.Stars)
# print(norm_stars.head())
# df.Stars = norm_stars
#
# print("Contributions Normalized")
# norm_contributions = normalize(df.Contributions)
# print(norm_contributions.head())
# df.Contributions = norm_contributions
#
# df.Weight = weight(df)
#
# print(df.head(10))
#
# #print("Length Dataframe: {}".format(df.size))
# # print(df)
# # print("DATAFRAME CREATED")
#
# df.to_csv('search_results_SORTED_DESCENDING_SIMILARITY_normalized_with_weight.csv')
# print("CSV CREATED")
#
#
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
# # print(gdf)
# # print("GEO DATAFRAME CREATED")
# #world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# # ax = world.plot(color='white', edgecolor='black')
# # gdf.plot(ax=ax, color='blue')
# # plt.show()
#
#
# # plt.savefig("map_images/map_plot_NEW_DESC.jpeg")
# # print("MAP 1 CREATED")
#
#
# # ax = world.plot(color='white', edgecolor='black')
# #
# # gdf.plot(ax=ax, color='red')
#
# #plt.show()
# #======================================================
#
# # similarity = np.asarray(df['Similarity'])
# # #scheme = mapclassify.Quantiles(similarity, k=3)
# #
# # fig = geoplot.choropleth(
# #         world, hue=similarity,
# #         cmap='Greens', figsize=(8, 4)
# #        )
# # fig.show()
#
# # data = [dict(type = 'choropleth',
# #             colorscale = 'Reds',
# #             locations=df['Country_Code'], # Spatial coordinates
# #             z = df['Similarity'].astype(float), # Data to be color-coded
# #             locationmode = 'USA-states', # set of locations match entries in `locations
# #             colorbar = {'title':"Cosine Similarity"},
# #            )]
# #
# # layout = dict(title = 'No Title',
# #               geo = dict(scope='usa', showlakes = True)) # limite map scope to USA)
# #
# # fig = dict( data=data, layout=layout )
# # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# # fig.show()
#
#
# path_to_file = 'custom.geo.json'
# with open(path_to_file) as f:
#     geo = geojson.load(f)
#
# fig = px.choropleth(data_frame=df,
#                     geojson=geo,
#                     locations='Country',
#                     locationmode = 'country names',
#                     color='Followers',
#                     color_continuous_scale='Viridis',
#                     range_color=(0, 1))
# #fig.show()
# fig.write_image("map_images/choropleth3_normalized_new.jpeg")
#----------------------------------------------------------------------------------

