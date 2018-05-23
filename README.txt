README
This code was written in python 3.6
Libraries:
- numpy 
- pandas
- sklearn
- scipy
- matplotlib
- argparse, zipfile, random, sys

Data Files:
- the raw YELP dataset can be found at https://www.kaggle.com/yelp-dataset/yelp-dataset
- cleaned files
	- (1,000 users) cleaned_data_1000.csv
	- (10,000 users) cleaned_data_10000.csv
	- (100,000 users 1) cleaned_data_0_100000.csv
	- (100,000 users 2) cleaned_data_200000_300000.csv
	- (100,000 users 3) cleaned_data_300000_400000.csv

Code files:
- preprocessing.py: this contains the functions to read in and clean the raw data. Running this as meain will attempt to
preprocess the entire yelp dataset in 100,000 user chunks. Note: the zip file from the Yelp dataset must be in the same folder.
- datavisualization.py: this contains the functions to visualize the four named fields from the preprocessed data. Running this
as main will display the graphs and mean/variance for the smallest dataset.
- clustering.py: this contains the functions to run Kmeans and DBASCAN on the cleaned data. Running this as main will cluster
and generate graphs using Kmeans for the three 100,000 user datasets and cluster and get graphs using DBSCAN for the 10,000 user
dataset.