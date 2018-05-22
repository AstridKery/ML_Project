import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generateHist(data, bins,title):

    plt.hist(data, bins, histtype='bar', rwidth=0.7)

    plt.xlabel("normalized scale")
    plt.ylabel("num users")
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
        tmp = (i - mean)**2

    variance = tmp/(len(x_input)-1)
    print(
        "Mean: {}, Variance: {}".format(mean, variance)
    )

    return mean, variance


def getStats(filename, data_file, title = "data"):
    columns = ['num_states','state_variance','1_day_max_reviews','star_variance']

    if len(filename) > 0:
        data_file = pd.read_csv(filename)

    results  = {}

    for named_col in columns:
        results[named_col] = getClusterStatistics(data_file[named_col])

        bins = np.sort(list(data_file[named_col].unique()))
        generateHist(data_file[named_col],bins, title+":"+named_col)

    print(results)




if __name__ == '__main__':
    getStats("cleaned_data_1000.csv",[])


