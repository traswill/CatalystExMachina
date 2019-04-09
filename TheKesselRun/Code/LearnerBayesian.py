# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Code Start: December 7, 2018

from TheKesselRun.Code.Plotter import Graphic
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation
from TheKesselRun.Code.LearnerOrder import SupervisedLearner, CatalystContainer

import pymc3 as pm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn import naive_bayes, decomposition

class BayesianLearner():
    def __init__(self):

        # Known Data
        self.X_dataset = None
        self.Y_dataset = None

        # Experimental design that is to be explored
        self.X_parameter_space = None

        self.learner = None


    def add_priors(self, X, Y):
        self.X_dataset = X
        self.Y_dataset = Y

    def gaussion_approximation(self):
        pass

    def set_learner(self, learner=None):
        if learner == 'GNB':
            self.learner = naive_bayes.GaussianNB(priors=None)
        else:
            print('No Learner Selected. Use set_learner method to establish a learner algorithm.')

    def apply_PCA(self, n_components=2):
        pca = decomposition.PCA(n_components=n_components)
        self.X_dataset = pca.fit_transform(self.X_dataset)
        # self.X_parameter_space = pca.fit_transform((self.X_parameter_space))

    def plot_PCA(self):
        df = pd.DataFrame(self.X_dataset, columns=['PCA1','PCA2','PCA3','PCA4','PCA5'])
        df['Y'] = self.Y_dataset
        df['Y'] = pd.qcut(df['Y'], q=3, labels=['Bad','Meh','Good'])
        sns.pairplot(data=df, hue='Y')
        plt.show()

    def train_learner(self):
        self.learner.fit(X=self.X_dataset, y=self.Y_dataset)

def load_nh3_catalysts(catcont, drop_empty_columns=True):
    """ Import NH3 data from Katie's HiTp dataset(cleaned). """
    df = pd.read_csv(r"..\Data\Processed\AllData_Condensed.csv", index_col=0)
    df.dropna(axis=0, inplace=True, how='all')

    # Drop RuK data that is inconsistent from file 5
    df.drop(index=df[df['ID'] == 20].index, inplace=True)
    df.drop(index=df[df['ID'] == 21].index, inplace=True)
    df.drop(index=df[df['ID'] == 22].index, inplace=True)
    # Using catalyst #24 for RuK (20ml)

    # Import Cl atoms during synthesis
    cl_atom_df = pd.read_excel(r'..\Data\Catalyst_Synthesis_Parameters.xlsx', index_col=0)

    # Loop through all data
    for index, dat in df.iterrows():
        # If the ID already exists in container, then only add an observation.  Else, generate a new catalyst.
        if dat['ID'] in catcont.catalyst_dictionary:
            catcont.catalyst_dictionary[dat['ID']].add_observation(
                temperature=dat['Temperature'],
                space_velocity=dat['Space Velocity'],
                gas=None,
                gas_concentration=dat['NH3'],
                pressure=None,
                reactor_number=int(dat['Reactor']),
                activity=dat['Conversion'],
                activity_error=dat['Standard Error'],
                selectivity=None
            )
        else:
            cat = CatalystObject()
            cat.ID = dat['ID']
            cat.add_element(dat['Ele1'], dat['Wt1'])
            cat.add_element(dat['Ele2'], dat['Wt2'])
            cat.add_element(dat['Ele3'], dat['Wt3'])
            cat.set_group(dat['Groups'])
            try:
                cat.add_n_cl_atoms(cl_atom_df.loc[dat['ID']].values[0])
            except KeyError:
                print('Catalyst {} didn\'t have Cl atoms'.format(cat.ID))
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()

            cat.add_observation(
                temperature=dat['Temperature'],
                space_velocity=dat['Space Velocity'],
                gas=None,
                gas_concentration=dat['NH3'],
                pressure=None,
                reactor_number=int(dat['Reactor']),
                activity=dat['Conversion'],
                activity_error=dat['Standard Error'],
                selectivity=None
            )

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

    catcont.build_master_container(drop_empty_columns=drop_empty_columns)

if __name__ == '__main__':
    # Load Known Dataset (output of v56 ERT algorithm)
    pth = r"..\Results\v56-remove-unfilled-features\result_dataset-v56-remove-unfilled-features-3-350orlessC.csv"
    known_df = pd.read_csv(pth, index_col=0)

    # Clean Dataset
    metalist = ['Name','Ele1','Load1','Ele2','Load2','Ele3','Load3']
    metadata_known_df = known_df[metalist].copy()
    measured_known_df = known_df[['Measured Conversion']].copy()
    predicted_known_df = known_df[['Predicted Conversion']].copy()
    features_known_df = known_df.drop(columns=metalist + ['Measured Conversion'] + ['Predicted Conversion'])

    # Generate Predictions from Model
    blearner = BayesianLearner()
    blearner.add_priors(features_known_df.values, measured_known_df['Measured Conversion'].values)
    blearner.set_learner(learner='GNB')
    # blearner.apply_PCA(n_components=5)
    # blearner.plot_PCA()
    blearner.train_learner()







