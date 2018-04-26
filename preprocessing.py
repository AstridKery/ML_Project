
'''
Created on Apr 25, 2018

@author: Pat
'''

import argparse
import zipfile
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def bag_of_words(review_frame):
    '''
    This function takes in reviews and generates a bag-of-words set of features per user
    :param review_frame: the dataframe with the reviews
    :return:
    '''
    vectorizer = CountVectorizer(stop_words='english')

    user_list = list(review_frame["user_id"].unique())
    user_text = {}
    all_text_list = []
    for user in user_list:
        user_frame = review_frame.loc[review_frame["user_id"] == user]
        all_text = " ".join(user_frame["text"].tolist())
        all_text = all_text.replace(".","")
        all_text = all_text.replace(",", "")
        all_text = all_text.replace(";", "")
        all_text = all_text.replace(":", "")
        all_text = all_text.replace("\n", " ")
        all_text = all_text.replace("\t", " ")
        all_text = all_text.replace("'", "")
        all_text = all_text.replace('"', "")
        all_text = all_text.replace("?", "")
        all_text = all_text.replace("!", "")
        all_text = all_text.replace("/", " ")
        all_text = all_text.replace("\\", " ")
        all_text = all_text.replace("  ", " ")
        all_text = all_text.lower()
        user_text[user] = all_text
        all_text_list.append(all_text)
    output = vectorizer.fit_transform(all_text_list)
    pca = PCA(n_components=50)
    lower_dim_stuff = pca.fit_transform(output.toarray())
    output_frame = pd.DataFrame(lower_dim_stuff)
    normalizer = MinMaxScaler()
    output_frame = pd.DataFrame(normalizer.fit_transform(output_frame))

    output_frame.index = user_list
    return output_frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp-dataset.zip',
                        help='Path to data file')
    return parser.parse_known_args()


if __name__ == "__main__":
    ARGS, unparsed = parse_args()

    with zipfile.ZipFile(ARGS.dataset, "r") as files:
        user = pd.read_csv(files.open('yelp_user.csv'), nrows = 1000)
        #print(user.head(n=15))
        #print("######################")
        reviews = pd.read_csv(files.open('yelp_review.csv'), nrows = 10000)
        #print(reviews.head(n=15))
        #print("#####################")
        #business = pd.read_csv(files.open('yelp_business.csv'), nrows = 1000)
        #print(business.head(n=15))

        the_bag = bag_of_words(review_frame=reviews)

        # business_attributes = pd.read_csv(files.open('yelp_business_attributes.csv'))
        # maybe we want business hours if we can somehow see if reviews were left
        # during business hours?
        # business_hours = pd.read_csv(files.open('yelp_business_hours.csv'))
        # check_in = pd.read_csv(files.open('yelp_checkin.csv'))

        # tip = pd.read_csv(files.open('yelp_tip.csv'))