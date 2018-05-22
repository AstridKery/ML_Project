'''
clustering.py - this file contains the function for running and analyzing clustering algorithms
'''

from datavisualization import *
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric

def find_best_K(dataframe):
    '''
    This function iteratively increases K as it runs K-means
    until the clusters do not appear to be stable enough
    :param dataframe: the preprocessed data
    :return: the best K value to use
    '''
    dataframe2 = dataframe.copy(deep = True)
    errors = []

    for K in range(3,20):
        print(K)
        #cluster the elements on K clusters
        assignments, centroids = run_Kmeans_clustering(dataframe2,K)
        assignment_frame = pd.DataFrame.from_dict(assignments,orient="index")
        total_frame = pd.concat([dataframe2,assignment_frame],axis=1)

        #print(assignment_frame)
        #get the Sum Squared Error for this clustering
        error = 0
        for cluster in centroids.keys():
            users = total_frame.loc[total_frame[0] == cluster].index.values.tolist()
            #get the data for the users that were put in this cluster
            dataframe_cluster = dataframe2.loc[users]

            #get the distances between the points and the centroid
            dataframe_cluster["dist"] = dataframe_cluster.apply(axis=1,func=lambda row: (distance.euclidean(row, centroids[cluster]))**2)

            #if this error has decreased sharply from the previous one, choose this K and stop
            error += np.sum(dataframe_cluster["dist"].tolist())

        errors.append(error)

    max_change = 0
    max_K = -1
    for K in range(4,20):
        change = abs(errors[K-3] - errors[K-4])
        print("The change between ", K - 3, "and", K - 4, "is", change)
        if change > max_change:
            max_K = K
            max_change = change
    print("found this best K",max_K)
    return max_K


def run_Kmeans_clustering(dataframe,K):
    '''
    This function runs the k-means algorithm to get the groups
    :param dataframe:
    :param K:
    :return:
    '''
    # run the k-means clustering on given data and K

    # if there is one giant cluster, do this again but take the big cluster out (might do this iteratively somehow?)
    done = False
    dataframe2 = dataframe.copy(deep = True)
    user_cluster_dict = {}
    centroid_data = {}
    extra_cluster = -1
    while not done:
        #print("Running kmeans algorithm ")

        clusterer = KMeans(n_clusters=K)
        clusterer.fit(dataframe2)

        #get the list of the labels
        labels = clusterer.labels_
        dataframe2["cluster"] = labels

        #get the number of elements in each cluster
        bins = np.bincount(labels)

        centroids = clusterer.cluster_centers_
        found = False
        for i in range(len(bins)):
            #once these get small, stop trying to seperate them
            if len(labels) < 100:
                break
            #print(bins[i]/len(labels))
            if bins[i]/len(labels) > 0.7:
                found = True
                #we have found a really large cluster, so record it then remove it
                #print("We found a really big cluster!:",bins[i],"/",len(labels))
                dataframe_cluster = dataframe2.loc[dataframe2["cluster"] == i]
                #assign them to something that will definitly not be used by kmeans
                for user in dataframe_cluster.index.values.tolist():
                    user_cluster_dict[user] = extra_cluster

                dataframe2 = dataframe2.loc[dataframe2["cluster"] != i]

                dataframe2.pop("cluster")
                centroid_data[extra_cluster] = centroids[i]
                extra_cluster -= 1
                break
        if not found:
            done = True
            user_cluster_dict = {**user_cluster_dict, **dataframe2["cluster"].to_dict()}
            dataframe2.pop("cluster")
            for i in range(len(centroids)):
                centroid_data[i] = centroids[i]

    return user_cluster_dict, centroid_data

def run_DBScan_clustering(dataframe, eps=0.03, min_samples=3):
    """
    This Function will run the DBSCAN algorithm to find the best clusters given the set parameters.

    :param dataframe: the data to fit to
    :return:
    """
    dataframe2 = dataframe.copy(deep=True)

    # this will exclude a column from the dataframe
    # cleaned_data.loc[:, cleaned_data.columns != 'Unnamed: 0']

    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(dataframe.loc[:, dataframe.columns != 'Unnamed: 0'])
    labels = clusterer.labels_

    user_list = dataframe.index.values.tolist()

    end_dict = {}
    for i in range(len(labels)):
        end_dict[user_list[i]] = labels[i]
    return end_dict

if __name__ == "__main__":
    for file in ["cleaned_data_0_100000.csv","cleaned_data_200000_300000.csv","cleaned_data_300000_400000.csv"]:
        print("LOOKING AT FILE",file)
        cleaned_data = pd.read_csv(file)
        cleaned_data.set_index("Unnamed: 0",inplace=True)

        best_K = 4

        #once you have the final groups, call Jose's graphing functions on each cluster to compare
        assignments, centroids = run_Kmeans_clustering(cleaned_data, best_K)
        assignment_frame = pd.DataFrame.from_dict(assignments, orient="index")
        total_frame = pd.concat([cleaned_data, assignment_frame], axis=1)

        for cluster in centroids.keys():
            print("looking at the centroids")
            users = total_frame.loc[total_frame[0] == cluster].index.values.tolist()
            # get the data for the users that were put in this cluster
            dataframe_cluster = cleaned_data.loc[users]
            getStats("",dataframe_cluster,"KMeans Cluster_"+str(cluster)+" ")

    cleaned_data = pd.read_csv("cleaned_data_10000.csv")
    cleaned_data.set_index("Unnamed: 0", inplace=True)
    print("running DBSCAN")
    assignments = run_DBScan_clustering(cleaned_data, eps=0.05, min_samples=125)
    print("Got the assignments for DBSCAN")
    assignment_frame = pd.DataFrame.from_dict(assignments, orient="index")
    total_frame = pd.concat([cleaned_data, assignment_frame], axis=1)
    print(total_frame)
    clusters = assignment_frame[0].unique()
    for cluster in clusters:
        cluster_frame = total_frame.loc[total_frame[0] == cluster]
        getStats("", cluster_frame, "DBSCAN Cluster_" + str(cluster) + " ")




