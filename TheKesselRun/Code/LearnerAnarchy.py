import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns
import os
import time

from TheKesselRun.Code.Catalyst import Catalyst

class Anarchy():
    def __init__(self):
        '''Initialize dictionary to hold import data'''
        self.catalyst_dataframe = pd.DataFrame()
        self.feature_set = pd.DataFrame()
        self.cluster_results = pd.DataFrame()

    def add_catalyst_dataframe(self, df):
        self.catalyst_dataframe = df

    #
    # def add_catalyst(self, index, catalyst):
    #     """ Add Catalysts to self.catalyst_dictionary.  This is the primary input function for the model. """
    #     base_index = index
    #     mod = 0
    #
    #     index = '{}_{}'.format(base_index, mod)
    #
    #     # Determine if key already exists in dictionary, modify key if so
    #     while index in self.catalyst_dictionary:
    #         mod += 1
    #         index = '{}_{}'.format(base_index, mod)
    #
    #     # Add to dictionary
    #     # self.catalyst_dictionary[index] = catalyst

    def build_feature_set(self, header):
        def load_catalysts_from_dataframe(input_df):
            eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)

            output_df = pd.DataFrame(index=input_df.index, columns=header)
            # print("Empty Dataframe: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            output_df.loc[input_df.index, input_df['E1']] = input_df['W1']
            # print("First Column: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            output_df.loc[input_df.index, input_df['E2']] = input_df['W2']
            # print("Second Column: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            output_df.loc[input_df.index, input_df['E3']] = input_df['W3']
            # print("Third Column: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            output_df.loc[input_df.index, input_df['E4']] = input_df['W4']
            # print("Fourth Column: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            output_df.loc[input_df.index, input_df['E5']] = input_df['W5']
            # print("Full Dataframe: {:03.2f} MB".format(output_df.memory_usage(deep=True).sum() / 1024 ** 2))

            working_df = output_df.dropna(axis=1, how='all')
            eledf = eledf.loc[working_df.columns]

            # print(working_df.columns)
            # print(eledf)
            print(input_df['E1'])
            print(eledf.loc[[input_df.loc[input_df.index, 'E1'].values,
                            input_df.loc[input_df.index, 'E2'].values,
                            input_df.loc[input_df.index, 'E3'].values,
                            input_df.loc[input_df.index, 'E4'].values,
                            input_df.loc[input_df.index, 'E5'].values]]
                )

            # output_df[input_df.index, eledf.columns] = eledf[[input_df['E1'], input_df['E2'], input_df['E3'], input_df['E4'], input_df['E5']]]

            return output_df

        df = load_catalysts_from_dataframe(self.catalyst_dataframe)


        #
        #
        # for catid, catobj in self.catalyst_dictionary.items():
        #     # Reset loading dictionary
        #     load_df = loading_df.copy()
        #
        #     # Add elements and loading to loading dict
        #     for ele, wt in catobj.elements.items():
        #         load_df.loc[catid, '{} Loading'.format(ele)] = wt / 100
        #
        #     # Create DF from features
        #     featdf = pd.DataFrame.from_dict(catobj.feature_dict, orient='index').transpose()
        #     featdf.index = [catid]
        #
        #     # Combine DFs
        #     df = pd.concat([load_df, featdf], axis=1)
        #     self.feature_set = pd.concat([self.feature_set, df], axis=0)
        #
        # self.feature_set.dropna(how='all', axis=1, inplace=True)
        # self.feature_set.fillna(value=0, inplace=True)

    def kmeans(self):
        alg = MiniBatchKMeans(n_clusters=64)
        alg.fit(self.feature_set.values)
        self.cluster_results = pd.DataFrame(alg.cluster_centers_, columns=self.feature_set.columns)
        self.cluster_results.to_csv('..\\Results\\Unsupervised\\Anarchy_Version_1_kmeans_res.csv')

    def find_closest_centroid(self):
        centroids_index = pairwise_distances_argmin(self.cluster_results, self.feature_set)
        res = self.feature_set.iloc[centroids_index].copy()
        res.to_csv('..\\Results\\Unsupervised\\Anarchy_Version_1.csv')