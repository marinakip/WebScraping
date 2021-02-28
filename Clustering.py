import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import re
import geojson
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
from sklearn.cluster import MeanShift


def cluster_and_plot(column, colors):
    clusters, cluster_labels, df_cluster_elements = clustering_by_feature(df, column)
    figure = create_map(df, column, clusters, colors, cluster_labels, df_cluster_elements)
    return figure, df_cluster_elements


def clustering_by_feature(df, column):
    sorted_df_by_column = sort_by_column(df, column)
    clusters, cluster_labels, df_cluster_elements = mean_shift_clustering(sorted_df_by_column, column)
    return clusters, cluster_labels, df_cluster_elements


def create_map(df, column, clusters, colors, cluster_labels, df_cluster_elements):
    # clusters = list(clusters)
    print("Clusters size: " + str(clusters))
    figure = px.scatter_mapbox(data_frame = df, lat = "Latitude", lon = "Longitude",
                               color = clusters.astype(str), size = column,
                               size_max = 30, zoom = 0.75, hover_name = 'Country',
                               title = 'Cluster by {}'.format(column),
                               hover_data = ['Followers', 'Following', 'Stars', 'Contributions', 'Url_profile'])
    figure.update_layout(mapbox_style = "carto-positron", legend_title_text = 'Cluster Number')
    return figure


def sort_by_column(df, column):
    sorted_df = df.sort_values(f'{column}')
    return sorted_df


def mean_shift_clustering(df, column):
    array = df[f'{column}'].to_numpy()
    X = np.reshape(array, (-1, 1))
    mean_shift = MeanShift(bandwidth = None, bin_seeding = True)
    mean_shift.fit(X)
    clusters = mean_shift.labels_
    centers = mean_shift.cluster_centers_
    # print("-----------------------")
    # print(column)
    # print("Centers")
    # print(centers)

    df['Cluster'] = clusters

    labels_unique = np.unique(clusters)
    print("Unique clusters:" + str(labels_unique))
    number_of_clusters = len(labels_unique)

    print("number of estimated clusters : %d" % number_of_clusters)
    print(clusters)
    cluster_labels = list(labels_unique)
    print("Cluster labels: " + str(cluster_labels))

    elements_in_clusters = get_cluster_elements(centers, clusters, column, df, number_of_clusters)

    df_cluster_elements = DataFrame(elements_in_clusters, columns = ['Column', 'Cluster Number', 'First Element',
                                                                     'Last Element', 'Cluster Centers'])
    print(df_cluster_elements['Cluster Centers'])
    return clusters, cluster_labels, df_cluster_elements


def get_cluster_elements(centers, clusters, column, df, number_of_clusters):
    clusters_dictionary = {i: np.where(clusters == i)[0] for i in range(number_of_clusters)}
    print(clusters_dictionary)
    elements_in_clusters = []
    for cluster_number in clusters_dictionary.keys():
        # print("Cluster: %d" % cluster_number)
        first = clusters_dictionary[cluster_number][0]
        last = clusters_dictionary[cluster_number][-1]
        # print("First: " + str(first))
        # print(("Last: " + str(last)))

        first_element = df[f'{column}'].iloc[first]
        last_element = df[f'{column}'].iloc[last]
        # print("First element: " + str(first_element))
        # print(("Last element: " + str(last_element)))
        # for i in range(len(centers)):
        center = centers[cluster_number]
        cluster_list_elements = [column, cluster_number, first_element, last_element, center]
        elements_in_clusters.append(cluster_list_elements)

    print('column, cluster_number, first_element, last_element, centers')
    print(elements_in_clusters)
    return elements_in_clusters


def clustering_auto(df):
    df.dropna(inplace = True)
    df_array = df[['Following', 'Followers', 'Stars', 'Contributions']].to_numpy()
    X = StandardScaler().fit_transform(df_array)
    kmeans = KMeans(init = "random", n_clusters = 5, n_init = 100, max_iter = 1000)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    # print(centers)
    # df['Cluster'] = cluster

    df['Cluster'] = clusters

    print("==================================================================")
    print("==================================================================")
    print("AUTO CLUSTERING")
    labels_unique = np.unique(clusters)
    print("Unique clusters:" + str(labels_unique))
    number_of_clusters = len(labels_unique)

    print("number of estimated clusters : %d" % number_of_clusters)
    print(clusters)
    cluster_labels = list(labels_unique)
    print("Cluster labels: " + str(cluster_labels))

    column = df[['Following', 'Followers', 'Stars', 'Contributions']]
    column_name = 'Following , Followers, Stars, Contributions'

    elements_in_clusters_auto = get_cluster_elements_auto(centers, clusters, column, df, number_of_clusters,
                                                          column_name)

    df_cluster_elements_auto = DataFrame(elements_in_clusters_auto, columns = ['Column', 'Cluster Number',
                                                                               'First Element',
                                                                               'Last Element', 'Cluster Centers'])
    print(df_cluster_elements_auto['Cluster Centers'])

    print("==================================================================")
    # print(df)
    figure = px.scatter_mapbox(data_frame = df, lat = "Latitude", lon = "Longitude",
                               color = df['Cluster'].astype(str), zoom = 0.75,
                               # color_continuous_scale = px.colors.cyclical.IceFire,
                               color_discrete_sequence = px.colors.qualitative.Bold,
                               hover_name = 'Country',
                               title = 'Auto Clustering',
                               hover_data = ['Followers', 'Following', 'Stars', 'Contributions', 'Url_profile'])
    figure.update_traces(mode = 'markers', marker_size = 12)
    figure.update_layout(mapbox_style = "carto-positron", legend_title_text = 'Cluster Number')
    return figure, df_cluster_elements_auto


def get_cluster_elements_auto(centers, clusters, column, df, number_of_clusters, column_name):
    clusters_dictionary = {i: np.where(clusters == i)[0] for i in range(number_of_clusters)}
    print(clusters_dictionary)
    elements_in_clusters = []
    for cluster_number in clusters_dictionary.keys():
        # print("Cluster: %d" % cluster_number)
        first = clusters_dictionary[cluster_number][0]
        last = clusters_dictionary[cluster_number][-1]
        # print("First: " + str(first))
        # print(("Last: " + str(last)))

        first_element = column.iloc[first]
        last_element = column.iloc[last]
        # print("First element: " + str(first_element))
        # print(("Last element: " + str(last_element)))
        # for i in range(len(centers)):
        center = centers[cluster_number]
        # column_name = 'Following, Followers, Stars, Contributions'
        cluster_list_elements = [column_name, cluster_number, first_element, last_element, center]
        elements_in_clusters.append(cluster_list_elements)

    print('column, cluster_number, first_element, last_element, centers')
    print(elements_in_clusters)
    return elements_in_clusters


def clustering_auto_weighted(df, followers_weight, following_weight, stars_weight, contributions_weight):
    # followers_weight = 0.3
    # following_weight = 2
    # stars_weight = 0.5
    # contributions_weight = 3

    df = weighted_dataframe(df, followers_weight, following_weight, stars_weight, contributions_weight)
    df.dropna(inplace = True)
    df_array = df[['Weighted_Following', 'Weighted_Followers',
                   'Weighted_Stars', 'Weighted_Contributions']].to_numpy()
    X = StandardScaler().fit_transform(df_array)
    kmeans = KMeans(init = "random", n_clusters = 5, n_init = 100, max_iter = 1000)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    # print(centers)
    df['Cluster_Weighted'] = clusters

    print("==================================================================")
    print("==================================================================")
    print("AUTO CLUSTERING")
    labels_unique = np.unique(clusters)
    print("Unique clusters:" + str(labels_unique))
    number_of_clusters = len(labels_unique)

    print("number of estimated clusters : %d" % number_of_clusters)
    print(clusters)
    cluster_labels = list(labels_unique)
    print("Cluster labels: " + str(cluster_labels))

    column = df[['Weighted_Following', 'Weighted_Followers', 'Weighted_Stars', 'Weighted_Contributions']]
    column_name = 'Weighted_Following, Weighted_Followers, Weighted_Stars, Weighted_Contributions'
    elements_in_clusters_auto = get_cluster_elements_auto(centers, clusters, column, df, number_of_clusters,
                                                          column_name)

    df_cluster_elements_auto_weighted = DataFrame(elements_in_clusters_auto,
                                                  columns = ['Column', 'Cluster Number', 'First Element',
                                                             'Last Element', 'Cluster Centers'])
    print(df_cluster_elements_auto_weighted['Cluster Centers'])

    print("==================================================================")

    # print(df)
    figure = px.scatter_mapbox(data_frame = df, lat = "Latitude", lon = "Longitude",
                               color = df['Cluster_Weighted'].astype(str), zoom = 0.75,
                               color_continuous_scale = px.colors.qualitative.D3,
                               hover_name = 'Country',
                               title = 'Auto Clustering Weighted',
                               hover_data = ['Followers', 'Following', 'Stars', 'Contributions', 'Url_profile'])
    figure.update_traces(mode = 'markers', marker_size = 12)
    figure.update_layout(mapbox_style = "carto-positron", legend_title_text = 'Cluster Info')
    return figure, df_cluster_elements_auto_weighted


def weighted_dataframe(df, followers_weight, following_weight, stars_weight, contributions_weight):
    weighted_followers = df['Followers'] * followers_weight
    weighted_following = df['Following'] * following_weight
    weighted_stars = df['Following'] * stars_weight
    weighted_contributions = df['Following'] * contributions_weight
    sum_of_weights = followers_weight + following_weight + stars_weight + contributions_weight
    df['Weighted_Followers'] = weighted_followers
    df['Weighted_Following'] = weighted_following
    df['Weighted_Stars'] = weighted_stars
    df['Weighted_Contributions'] = weighted_contributions
    return df


def cluster_and_plot_weighted(column, colors):
    followers_weight = 0.3
    following_weight = 2
    stars_weight = 0.5
    contributions_weight = 3

    df_weighted = weighted_dataframe(df, followers_weight, following_weight, stars_weight, contributions_weight)
    clusters_weighted, cluster_labels, df_cluster_elements = clustering_by_feature(df_weighted, column)
    figure_weighted = create_map(df, column, clusters_weighted, colors, cluster_labels, df_cluster_elements)
    return figure_weighted, df_cluster_elements


def elbow_method(df):
    df.dropna(inplace = True)
    df_array = df[['Following', 'Followers', 'Stars', 'Contributions']].to_numpy()
    X = StandardScaler().fit_transform(df_array)
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters = k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


# def clustering(df, column):
#     df.dropna(inplace = True)
#     df_array = df[f'{column}'].to_numpy()
#     reshaped = np.array(df_array).reshape((len(df_array), 1))
#     X = StandardScaler().fit_transform(reshaped)
#     kmeans = KMeans(init = "random", n_clusters = 5, n_init = 100, max_iter = 1000)
#     model = kmeans.fit(X)
#     cluster = kmeans.predict(X)
#     #cluster = kmeans.fit_predict(reshaped)
#     df['Cluster'] = cluster
#     centroids = kmeans.cluster_centers_
#     # print(df)
#     # print("CLUSTER CENTERS")
#     # print(centroids)
#     figure = px.scatter_mapbox(data_frame = df, lat = "Latitude", lon = "Longitude",
#                                color = "Cluster", size = "Cluster",
#                                color_continuous_scale = px.colors.cyclical.IceFire,
#                                size_max = 70, zoom = 0.75, hover_name = 'Country',
#                                hover_data = ['Followers', 'Stars', 'Contributions', 'Url_profile'])
#     figure.update_layout(mapbox_style = "carto-positron")
#     # plt.xlabel('Cluster')
#     # plt.ylabel(f'{column}')
#     # figure = sns.barplot(data = df, x = 'Cluster', y = f'{column}')
#     #
#     # name = str(column)
#     # name = re.sub(r'\\', ' ', name)
#     #
#     # print(name)
#     #plt.show()
#     #plt.savefig("map_images/clusters_barchart_{}_kmeans_with_arguments_normalized.jpeg".format(name))
#     return figure


# def clustering_with_weight(df, column):
#     print("inside clustering")
#     df.dropna(inplace = True)
#     kmeans = KMeans(init = "random", n_clusters = 10, n_init = 100, max_iter = 1000)
#     print("kmeans ok")
#     cluster = kmeans.fit_predict(df[[f'{column}', 'Weight']])
#     print("cluster ok")
#     df['Cluster'] = cluster
#     centroids = kmeans.cluster_centers_
#     print(df.head(5))
#     # print("CLUSTER CENTERS")
#     # print(centroids)
#     ##plt.xlabel(f'{column}')
#     ##plt.ylabel('Weight')
#     print("before cluster figure ok")
#
#     path_to_file = 'custom.geo.json'
#     with open(path_to_file) as f:
#         geo = geojson.load(f)
#
#     # figure = px.choropleth(data_frame = df,
#     #                     geojson = geo,
#     #                     locations = 'Country',
#     #                     locationmode = 'country names',
#     #                     color = 'Cluster',
#     #                     color_continuous_scale = 'Viridis',
#     #                     range_color = (0, 10))
#
#     figure = px.scatter_mapbox(data_frame = df, lat = "Latitude", lon = "Longitude",
#                             color = "Cluster", size = "Cluster",
#                             color_continuous_scale = px.colors.cyclical.IceFire,
#                             size_max = 70, zoom = 0.75, hover_name = 'Country',
#                             hover_data = ['Followers', 'Stars', 'Contributions', 'Url_profile'])
#     figure.update_layout(mapbox_style = "carto-positron")
#     ##figure = sns.scatterplot(data = df, x =f'{column}', y ='Weight')
#     print("cluster figure end ok")
#     name = str(column)
#     name = re.sub(r'\\', ' ', name)
#     # print(name)
#     #plt.show()
#     #plt.savefig("map_images/clusters_with_weights_{}_kmeans_normalized.jpeg".format(name))
#     return figure

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY.csv")
# df.dropna(inplace = True)
# # elbow_method(df)
# figure_auto, df_cluster_elements_auto = clustering_auto(df)
#
# followers_weight = 0.3
# following_weight = 2
# stars_weight = 0.5
# contributions_weight = 3
# figure_auto_weighted, df_cluster_elements_auto_weighted = clustering_auto_weighted(df, followers_weight,
#                                                                                    following_weight, stars_weight,
#                                                                                    contributions_weight)
#
# figure, df_cluster_elements = cluster_and_plot('Following', px.colors.carto.Magenta)
# figure_weighted, df_cluster_elements_weighted = cluster_and_plot_weighted('Weighted_Following',
#                                                                           px.colors.qualitative.Pastel)
#
# figure2, df_cluster_elements2 = cluster_and_plot('Followers', px.colors.qualitative.Antique)
# figure_weighted2, df_cluster_elements_weighted2 = cluster_and_plot_weighted('Weighted_Followers',
#                                                                             px.colors.sequential.Cividis)
#
# figure3, df_cluster_elements3 = cluster_and_plot('Stars', px.colors.qualitative.Alphabet)
# figure_weighted3, df_cluster_elements_weighted3 = cluster_and_plot_weighted('Weighted_Stars',
#                                                                             px.colors.qualitative.Prism)
#
# figure4, df_cluster_elements4 = cluster_and_plot('Contributions', px.colors.sequential.Magenta)
# figure_weighted4, df_cluster_elements_weighted4 = cluster_and_plot_weighted('Weighted_Contributions',
#                                                                             px.colors.qualitative.Prism)
#
# app = dash.Dash(__name__)
#
# app.layout = html.Div([
#     html.Div(dcc.Graph(id = "graph", figure = figure)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table", columns = [{'name': i, 'id': i} for i in df_cluster_elements.columns],
#                                   data = df_cluster_elements.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph2", figure = figure_weighted)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table2",
#                                   columns = [{'name': i, 'id': i} for i in df_cluster_elements_weighted.columns],
#                                   data = df_cluster_elements_weighted.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph3", figure = figure2)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table3", columns = [{'name': i, 'id': i} for i in df_cluster_elements2.columns],
#                                   data = df_cluster_elements2.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph4", figure = figure_weighted2)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table4",
#                                   columns = [{'name': i, 'id': i} for i in df_cluster_elements_weighted2.columns],
#                                   data = df_cluster_elements_weighted2.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph5", figure = figure3)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table5", columns = [{'name': i, 'id': i} for i in df_cluster_elements3.columns],
#                                   data = df_cluster_elements3.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph6", figure = figure_weighted3)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table6",
#                                   columns = [{'name': i, 'id': i} for i in df_cluster_elements_weighted3.columns],
#                                   data = df_cluster_elements_weighted3.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph7", figure = figure4)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table7", columns = [{'name': i, 'id': i} for i in df_cluster_elements4.columns],
#                                   data = df_cluster_elements4.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph8", figure = figure_weighted4)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table8",
#                                   columns = [{'name': i, 'id': i} for i in df_cluster_elements_weighted4.columns],
#                                   data = df_cluster_elements_weighted4.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph9", figure = figure_auto)),
#     html.Br(),
#     html.Br(),
#     html.Div(
#         dash_table.DataTable(id = "table9", columns = [{'name': i, 'id': i} for i in df_cluster_elements_auto.columns],
#                              data = df_cluster_elements_auto.to_dict('records'))),
#
#     html.Br(),
#     html.Br(),
#     html.Div(dcc.Graph(id = "graph10", figure = figure_auto_weighted)),
#     html.Br(),
#     html.Br(),
#     html.Div(dash_table.DataTable(id = "table10",
#                                   columns = [{'name': i, 'id': i} for i in df_cluster_elements_auto_weighted.columns],
#                                   data = df_cluster_elements_auto_weighted.to_dict('records'))),
#
# ])
#
# app.run_server(debug = True)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY_normalized.csv")
######+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY_normalized_with_weight.csv")
#####clustering_with_weight2new(df)
# df.dropna(inplace = True)
# data = df[['Following', 'Followers', 'Stars', 'Contributions']]
# data = df['Following']
# print(data.head(5))
##X = StandardScaler().fit_transform(data)
# print(X)
# elbow_method(X)
##clustering_by_feature(X)

#
# #df = df[['Followers', 'Following']]
# df.dropna(inplace=True)
# clustering(df, 'Followers')
# clustering(df, 'Following')
# clustering(df, 'Stars')
# clustering(df, 'Contributions')
# clustering(df, 'Weight')
# clustering_with_weight(df, 'Followers')
# clustering_with_weight(df, 'Following')
# clustering_with_weight(df, 'Stars')
# clustering_with_weight(df, 'Contributions')
# clustering_with_weight(df,  'Weight')
# # followers_array = df['Followers'].to_numpy()
# # temp = np.array(followers_array).reshape((len(followers_array), 1))
# # temp = scaler.transform(temp)
# # folloers.reshape(-1, 1)
# ######elbow_method(temp)
# # print("DATAFRAME")
# #print(df)
# #print(df.head(5))
#
# #============================================================================
# #kmeans = KMeans(init="random", n_clusters=3, n_init=100, max_iter=1000)
# kmeans = KMeans(init="random", n_clusters=10, n_init=100, max_iter=1000)
#
# # #kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, max_iter=300, random_state=42)
# #kmeans.fit(scaled_df)
# #kmeans.fit(df)
#
# #cluster = kmeans.fit_predict(df[['Followers', 'Weight']])
# cluster = kmeans.fit_predict(temp)
#
# df['Cluster'] = cluster
# #print(cluster)
# print(df)
# # print("SSE")
# # sse = kmeans.inertia_
# # print(sse)
# # print("CLUSTER CENTERS")
# # centroids = kmeans.cluster_centers_
# # print(centroids)
# #========================================================================
#
# centroids = kmeans.cluster_centers_
# print("CLUSTER CENTERS")
# print(centroids)
# # cluster_array = df['Cluster'].to_numpy()
# # sb.violinplot(sb.violinplot(x=cluster_array, y=followers_array, data=df))
# #clusters_plot = plt.scatter(df['Followers'], df['Weight'], c= kmeans.labels_.astype(float), s=50, label = 'Clusters')
# #clusters_plot = plt.scatter(df['Followers'], df['Cluster'], c= kmeans.labels_.astype(float), s=50, label = 'Clusters')
# sns.barplot(data = df, x='Cluster', y='Followers')
# plt.show()
# centers_plot = plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, label = 'Centers')
#
# #plt.title('Clusters Followers vs Weight')
# plt.xlabel('Followers')
# plt.ylabel('Weight')
# plt.legend(handles=[clusters_plot, centers_plot])
# plt.show()
# plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_with_arguments_normalized.jpeg")
# #plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments.jpeg")

#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
