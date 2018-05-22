import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generateHist(data, bins, title, xlabel, ylabel):

    plt.hist(data, bins, histtype='bar', rwidth=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def getClusterStatistics(x_input):
    total = 0
    for i in x_input:
        total += i
    mean = total /len(x_input)

    tmp = 0
    for i in x_input:
        tmp = (i - mean) ** 2

    variance = tmp/(len(x_input)-1)
    print(
        "Mean: {}, Variance: {}".format(mean, variance)
    )

    return mean, variance


def getStats(filename, data_file, title = "data"):

    if len(filename) > 0:
        data_file = pd.read_csv(filename)

    results  = {}

    dataset = "Dataset 1"

    col_name = 'num_states'
    results[col_name] = getClusterStatistics(data_file[col_name])
    bins = np.sort(list(data_file[col_name].unique()))
    generateHist(data_file[col_name], bins, "Number of States, " + dataset, "Number of States Normalized", "Number of Users")
    
    col_name = 'state_variance'
    results[col_name] = getClusterStatistics(data_file[col_name])
    bins = np.sort(list(data_file[col_name].unique()))
    generateHist(data_file[col_name], bins, "State Variance, " + dataset, "Variance Normalized", "Number of Users")
    
    col_name = '1_day_max_reviews'
    results[col_name] = getClusterStatistics(data_file[col_name])
    bins = np.sort(list(data_file[col_name].unique()))
    generateHist(data_file[col_name], bins, "Max Reviews in a Day, " + dataset, "Number of Reviews Normalized", "Number of Users")
    
    col_name = 'star_variance'
    results[col_name] = getClusterStatistics(data_file[col_name])
    bins = np.sort(list(data_file[col_name].unique()))
    generateHist(data_file[col_name], bins, "Star Variance, " + dataset, "Number of Stars Normalized", "Number of Users")

    print(results)

#def getDBSCANStatitstics():


if __name__ == '__main__':
    getStats("cleaned_data_1000.csv",[])


