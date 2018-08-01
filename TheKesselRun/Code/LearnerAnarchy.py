import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns
import os
import time

from TheKesselRun.Code.Catalyst import CatalystObject

class Anarchy():
    def __init__(self):
        '''Initialize dictionary to hold import data'''
        self.catalyst_dataframe = pd.DataFrame()
        self.features = pd.DataFrame()
        self.cluster_results = pd.DataFrame()
        self.index = 0

        self.loading_dataframe = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation'],
                                 index_col='Abbreviation').transpose()
        self.loading_dataframe.columns = ['{} Loading'.format(ele) for ele in self.loading_dataframe.columns]

    def set_catalyst_dataframe(self, df):
        self.catalyst_dataframe = df

    def add_catalyst(self, index, catalyst):
        """ Add Catalysts to self.catalyst_dictionary.  This is the primary input function for the model. """
        base_index = index
        mod = 0

        index = '{}_{}'.format(base_index, mod)

        # Determine if key already exists in dictionary, modify key if so
        while index in self.catalyst_dictionary:
            mod += 1
            index = '{}_{}'.format(base_index, mod)

        # Add to dictionary
        self.catalyst_dictionary[index] = catalyst

    def build_feature_set(self):
        output_df = pd.DataFrame()

        for index, row in self.catalyst_dataframe.iterrows():
            cat = CatalystObject()
            cat.add_element(row['E1'], row['W1'])
            cat.add_element(row['E2'], row['W2'])
            cat.add_element(row['E3'], row['W3'])
            cat.add_element(row['E4'], row['W4'])
            cat.add_element(row['E5'], row['W5'])
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_unsupervised_properties()

            # Reset loading dictionary
            load_df = self.loading_dataframe.copy()
            for ele, wt in cat.elements.items():
                load_df.loc[index, '{} Loading'.format(ele)] = wt / 100

            featdf = pd.DataFrame.from_dict(cat.feature_dict, orient='index').transpose()
            featdf.index = [index]

            df = pd.concat([load_df, featdf], axis=1)
            output_df = pd.concat([output_df, df], axis=0)

        self.catalyst_dataframe = output_df

        # Drop columns with all NaNs and catalysts that contain 0 elements
        self.catalyst_dataframe.dropna(how='all', axis=1, inplace=True)
        self.catalyst_dataframe.fillna(value=0, inplace=True)
        # self.features = self.catalyst_dataframe.drop(columns=['E1','E2','E3','E4','E5','W1','W2','W3','W4','W5'])
        self.features = self.catalyst_dataframe.copy()

    def kmeans(self, sv=None):
        alg = MiniBatchKMeans(n_clusters=64, compute_labels=False)
        alg.fit(self.features.values)
        self.cluster_results = pd.DataFrame(alg.cluster_centers_, columns=self.features.columns)
        if sv is None:
            self.cluster_results.to_csv('..\\Results\\Unsupervised\\Anarchy_Batch {}_kmeans_res.csv'.format(self.index))
        else:
            self.cluster_results.to_csv(sv)

    def find_closest_centroid(self, sv=None):
        centroids_index = pairwise_distances_argmin(self.cluster_results, self.features)
        res = self.features.iloc[centroids_index].copy()
        if sv is None:
            res.to_csv('..\\Results\\Unsupervised\\Anarchy_Batch {}_kmedian_res.csv'.format(self.index))
        else:
            res.to_csv(sv)