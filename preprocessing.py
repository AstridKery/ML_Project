
'''
Created on Apr 25, 2018

@author: Pat
'''

import argparse
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def other_info_about_reviews(review_frame, business_frame):
    '''
    This function generates the other features we are pulling out of the data
    :param review_frame: the dataframe with the reviews
    :param business_frame: the dataframe with the business information
    :return:
    '''

    user_list = list(review_frame["user_id"].unique())
    data_list = []
    for user in user_list:
        user_dict = {"num_states":0,"variance":0, "1_day_reviews":0, "star_variance":0}
        user_frame = review_frame.loc[review_frame["user_id"] == user].copy(deep = True)

        #get the state from the business dataframe and add it to the review frame
        user_frame["state"] = user_frame["business_id"].apply(lambda x: business_frame.loc[business_frame["business_id"] == x]["state"].tolist()[0])

        #get the total number of states they have reviewed in
        user_dict["num_states"] = len(user_frame["state"].unique())

        #get the binary columns for the states
        dummies = pd.get_dummies(user_frame["state"])

        #takes the average of the variances of the different binary  columns
        variance_list = []
        for col in dummies:
            variance_list.append(np.var(dummies[col].tolist()))
        user_dict["variance"] = np.average(variance_list)
        data_list.append(user_dict)

        #get the max number of reviews left in the same day
        day_counts = np.max(user_frame["date"].value_counts())
        user_dict["1_day_reviews"] = day_counts

        #get the variance for the number of stars they give a business
        user_dict["star_variance"] = np.var(user_frame["stars"].tolist())

    data_frame = pd.DataFrame(data_list)
    normalizer = MinMaxScaler()
    #normalize everything between 0 and 1
    output_frame = pd.DataFrame(normalizer.fit_transform(data_frame))

    #label the rows and columns with usernames and the columns we just made
    output_frame.index = user_list
    output_frame.columns = ["num_states","state_variance", "1_day_max_reviews", "star_variance"]
    return output_frame

def bag_of_words(review_frame, pca_count = 50):
    '''
    This function takes in reviews and generates a bag-of-words set of features per user
    :param review_frame: the dataframe with the reviews
    :return:
    '''
    vectorizer = CountVectorizer(stop_words='english')

    user_list = list(review_frame["user_id"].unique())
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
        all_text_list.append(all_text)
    output = vectorizer.fit_transform(all_text_list)
    pca = PCA(n_components=pca_count)
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
        business = pd.read_csv(files.open('yelp_business.csv'))
        #print(business.head(n=50))

        the_bag = bag_of_words(review_frame=reviews)
        state_info = other_info_about_reviews(review_frame=reviews,business_frame=business)

        total_prep_so_far = pd.concat([the_bag,state_info], axis=1)
        print(total_prep_so_far)

        # business_attributes = pd.read_csv(files.open('yelp_business_attributes.csv'))
        # maybe we want business hours if we can somehow see if reviews were left
        # during business hours?
        # business_hours = pd.read_csv(files.open('yelp_business_hours.csv'))
        # check_in = pd.read_csv(files.open('yelp_checkin.csv'))

        # tip = pd.read_csv(files.open('yelp_tip.csv'))