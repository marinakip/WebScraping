import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# def clustering_kmeans(df, column1, column2):
#     df.dropna(inplace = True)
#     scaler = StandardScaler()
#     scaled_df = scaler.fit_transform(df)
#     kmeans = KMeans(n_clusters = 3).fit(df)
#     centroids = kmeans.cluster_centers_
#     print("CLUSTER CENTERS")
#     print(centroids)
#     plt.scatter(column1, column2, c = kmeans.labels_.astype(float), s = 50, alpha = 0.5)
#     plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 50)
#     plt.xlabel('Latitude')
#     plt.ylabel('Longitude')
#     # plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments.jpeg")
#     plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments_df_new.jpeg")
#     plt.show()

#df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY.csv")
df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY_normalized.csv")
df = df[['Latitude', 'Longitude']]
column1 = df['Latitude']
column2 = df['Longitude']
#clustering_kmeans(df, column1, column2)
#df = df[['Followers', 'Following']]
df.dropna(inplace=True)
# print("DATAFRAME")
#print(df)
print(df.head(5))
# scaler = StandardScaler()
# scaled_df = scaler.fit_transform(df)
# print("SCALED DATAFRAME")
# print(scaled_df[:5])
#============================================================================
kmeans = KMeans(init="random", n_clusters=3, n_init=100, max_iter=1000)
# #kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, max_iter=300, random_state=42)
#kmeans.fit(scaled_df)
kmeans.fit(df)
#
# print("SSE")
# sse = kmeans.inertia_
# print(sse)
# print("CLUSTER CENTERS")
# centroids = kmeans.cluster_centers_
# print(centroids)
#========================================================================

# kmeans = KMeans(n_clusters = 3).fit(df)
# # kmeans = KMeans(n_clusters = 3).fit(scaled_df)
centroids = kmeans.cluster_centers_
# print("CLUSTER CENTERS")
# print(centroids)
plt.scatter(df['Latitude'], df['Longitude'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_with_arguments_normalized.jpeg")
#plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments.jpeg")
# plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments_scaled_df.jpeg")
# plt.show()
# #plt.scatter(df['Followers'], df['Following'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#
# #======================================================================================
# # plt.scatter(df['Latitude'], df['Longitude'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
# # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# # #plt.savefig("map_images/clusters_plot_latitude_longitude.jpeg")
# # #plt.savefig("map_images/clusters_plot_latitude_longitude_kmeansplusplus.jpeg")
# # plt.show()
# #======================================================================================







