import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generateHist(data, bins):

    plt.hist(data, bins, histtype='bar', rwidth=0.7)

    plt.xlabel("data")
    plt.ylabel("bins")
    plt.title("Data visiualization")
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


def main():
    columns = ['num_states','state_variance','1_day_max_reviews','star_variance']


    data_file = pd.read_csv("cleaned_data_1000.csv")

    results  = {}

    for named_col in columns:
        results[named_col] = getClusterStatistics(data_file[named_col])

        bins = np.sort(list(data_file[named_col].unique()))
        generateHist(data_file[named_col],bins)

    print(results)




if __name__ == '__main__':
    main()


