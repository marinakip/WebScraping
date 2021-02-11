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
import seaborn as sns
import re


def elbow_method(array):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(array)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def clustering(df, column):
    df_array = df[f'{column}'].to_numpy()
    reshaped = np.array(df_array).reshape((len(df_array), 1))
    kmeans = KMeans(init = "random", n_clusters = 10, n_init = 100, max_iter = 1000)
    cluster = kmeans.fit_predict(reshaped)
    df['Cluster'] = cluster
    centroids = kmeans.cluster_centers_
    # print(df)
    # print("CLUSTER CENTERS")
    # print(centroids)
    plt.xlabel('Cluster')
    plt.ylabel(f'{column}')
    figure = sns.barplot(data = df, x = 'Cluster', y = f'{column}')

    name = str(column)
    name = re.sub(r'\\', ' ', name)
    # print(name)
    #plt.show()
    #plt.savefig("map_images/clusters_barchart_{}_kmeans_with_arguments_normalized.jpeg".format(name))
    return figure

def clustering_with_weight(df, column):
    kmeans = KMeans(init = "random", n_clusters = 10, n_init = 100, max_iter = 1000)
    cluster = kmeans.fit_predict(df[[f'{column}', 'Weight']])
    df['Cluster'] = cluster
    centroids = kmeans.cluster_centers_
    # print(df)
    # print("CLUSTER CENTERS")
    # print(centroids)
    plt.xlabel(f'{column}')
    plt.ylabel('Weight')
    figure = sns.scatterplot(data = df, x =f'{column}', y ='Weight')
    name = str(column)
    name = re.sub(r'\\', ' ', name)
    # print(name)
    #plt.show()
    #plt.savefig("map_images/clusters_with_weights_{}_kmeans_normalized.jpeg".format(name))
    return figure


#df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY.csv")
#df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY_normalized.csv")

df = pd.read_csv("search_results_SORTED_DESCENDING_SIMILARITY_normalized_with_weight.csv")


#df = df[['Followers', 'Following']]
df.dropna(inplace=True)
clustering(df, 'Followers')
clustering(df, 'Following')
clustering(df, 'Stars')
clustering(df, 'Contributions')
clustering(df, 'Weight')
clustering_with_weight(df, 'Followers')
clustering_with_weight(df, 'Following')
clustering_with_weight(df, 'Stars')
clustering_with_weight(df, 'Contributions')
clustering_with_weight(df,  'Weight')
# followers_array = df['Followers'].to_numpy()
# temp = np.array(followers_array).reshape((len(followers_array), 1))
# temp = scaler.transform(temp)
# folloers.reshape(-1, 1)
######elbow_method(temp)
# print("DATAFRAME")
#print(df)
#print(df.head(5))

#============================================================================
#kmeans = KMeans(init="random", n_clusters=3, n_init=100, max_iter=1000)
kmeans = KMeans(init="random", n_clusters=10, n_init=100, max_iter=1000)

# #kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, max_iter=300, random_state=42)
#kmeans.fit(scaled_df)
#kmeans.fit(df)

#cluster = kmeans.fit_predict(df[['Followers', 'Weight']])
cluster = kmeans.fit_predict(temp)

df['Cluster'] = cluster
#print(cluster)
print(df)
# print("SSE")
# sse = kmeans.inertia_
# print(sse)
# print("CLUSTER CENTERS")
# centroids = kmeans.cluster_centers_
# print(centroids)
#========================================================================

centroids = kmeans.cluster_centers_
print("CLUSTER CENTERS")
print(centroids)
# cluster_array = df['Cluster'].to_numpy()
# sb.violinplot(sb.violinplot(x=cluster_array, y=followers_array, data=df))
#clusters_plot = plt.scatter(df['Followers'], df['Weight'], c= kmeans.labels_.astype(float), s=50, label = 'Clusters')
#clusters_plot = plt.scatter(df['Followers'], df['Cluster'], c= kmeans.labels_.astype(float), s=50, label = 'Clusters')
sns.barplot(data = df, x='Cluster', y='Followers')
plt.show()
centers_plot = plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, label = 'Centers')

#plt.title('Clusters Followers vs Weight')
plt.xlabel('Followers')
plt.ylabel('Weight')
plt.legend(handles=[clusters_plot, centers_plot])
plt.show()
plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_with_arguments_normalized.jpeg")
#plt.savefig("map_images/clusters_plot_latitude_longitude_kmeans_no_arguments.jpeg")








