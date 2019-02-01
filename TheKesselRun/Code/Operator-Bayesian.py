# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: January 28, 2019

from TheKesselRun.Code.LearnerOrder import SupervisedLearner, CatalystContainer
from TheKesselRun.Code.LearnerAnarchy import Anarchy
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation
from TheKesselRun.Code.Plotter import Graphic

from sklearn.metrics import r2_score, explained_variance_score, \
        mean_absolute_error, roc_curve, recall_score, precision_score, mean_squared_error, accuracy_score

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import time
import scipy.cluster
import glob
import random
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
                selectivity=None
            )
        else:
            cat = CatalystObject()
            cat.ID = dat['ID']
            cat.add_element(dat['Ele1'], dat['Wt1'])
            cat.add_element(dat['Ele2'], dat['Wt2'])
            cat.add_element(dat['Ele3'], dat['Wt3'])
            cat.input_group(dat['Groups'])
            try:
                cat.input_n_cl_atoms(cl_atom_df.loc[dat['ID']].values[0])
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
                selectivity=None
            )

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

    catcont.build_master_container(drop_empty_columns=drop_empty_columns)

def load_skynet(version, drop_loads=False, drop_na_columns=True, ru_filter=0):
    # Load Data
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer, drop_empty_columns=drop_na_columns)

    # Init Learner
    skynet = SupervisedLearner(version=version)
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=ru_filter,
        pressure_filter=None
    )

    # Set algorithm and add data
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_static_dataset(catalyst_container=catcontainer)

    # Set parameters
    skynet.set_target_columns(cols=['Measured Conversion'])
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID'])

    # Lists for dropping certain features
    zpp_list = ['Zunger Pseudopotential (d)', 'Zunger Pseudopotential (p)',
                'Zunger Pseudopotential (pi)', 'Zunger Pseudopotential (s)',
                'Zunger Pseudopotential (sigma)']

    if drop_loads:
        load_list = ['{} Loading'.format(x) for x in
                     ['Ru', 'Cu', 'Y', 'Mg', 'Mn',
                      'Ni', 'Cr', 'W', 'Ca', 'Hf',
                      'Sc', 'Zn', 'Sr', 'Bi', 'Pd',
                      'Mo', 'In', 'Rh', 'K']]
    else:
        load_list = []

    skynet.set_drop_columns(
        cols=['reactor', 'Periodic Table Column', 'Mendeleev Number', 'Norskov d-band', 'n_Cl_atoms']
                                 + zpp_list + ['Number Unfilled Electrons', 'Number s-shell Unfilled Electrons',
                                               'Number p-shell Unfilled Electrons', 'Number d-shell Unfilled Electrons',
                                               'Number f-shell Unfilled Electrons'])

    # skynet.set_drop_columns(cols=['reactor', 'Periodic Table Column', 'Mendeleev Number', 'Norskov d-band', 'n_Cl_atoms']
    #                              + zpp_list + load_list)

    skynet.filter_static_dataset()
    return skynet

def predict_design_space(version):
    skynet = load_skynet(version=version, ru_filter=0, drop_na_columns=False)
    skynet.set_learner('etr', 'etr-uncertainty')
    skynet.set_filters(temperature_filter='350orless')
    skynet.filter_static_dataset()
    skynet.train_data()
    skynet.calculate_tau()
    skynet.predict_crossvalidate(kfold=10)
    skynet.evaluate_regression_learner()

    # init catcont
    catcont = CatalystContainer()

    # Setup element-promoter-loading combinations
    ch_eles = ['Y','Hf','Mg','Ca','Sr']
    ch_prom = ['K','Na','Cs','Ba']
    prom_load = [5, 10, 15, 20, 25]

    # loop through all combinations
    for ele in ch_eles:
        for prom in ch_prom:
            for load in prom_load:
                cat = CatalystObject()
                cat.ID = 'A_{}'.format('{}-{}-{}'.format(ele, prom, load))
                cat.add_element('Ru', 3)
                cat.add_element(ele, 1)
                cat.add_element(prom, load)
                cat.input_group(-1)
                cat.feature_add_n_elements()
                cat.feature_add_Lp_norms()
                cat.feature_add_elemental_properties()

                cat.add_observation(
                    temperature=250,
                    space_velocity=2000,
                    gas_concentration=1,
                    reactor_number=0
                )

                cat.add_observation(
                    temperature=300,
                    space_velocity=2000,
                    gas_concentration=1,
                    reactor_number=0
                )

                cat.add_observation(
                    temperature=350,
                    space_velocity=2000,
                    gas_concentration=1,
                    reactor_number=0
                )

                catcont.add_catalyst(index=cat.ID, catalyst=cat)
    catcont.build_master_container(drop_empty_columns=False)
    skynet.load_static_dataset(catcont)

    skynet.set_training_data()
    skynet.predict_data()
    skynet.calculate_uncertainty()
    skynet.compile_results(sv=True)
    return skynet

def get_drop_colunms(ru_filter=0):
    skynet = load_skynet(version=version)
    skynet.set_filters(
        element_filter=3,
        temperature_filter=300,
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=ru_filter,
        pressure_filter=None
    )

    # Use the full dataset to calculate the columns to keep, then drop all other features
    skynet.filter_static_dataset()
    feat_selector = SelectKBest(score_func=f_regression, k=20)
    feats = feat_selector.fit_transform(skynet.features, skynet.labels)
    feats = feat_selector.inverse_transform(feats)
    skynet.features_df[:] = feats
    kbest_column_list = list(skynet.features_df.loc[:, skynet.features_df.sum(axis=0) != 0].columns)
    return kbest_column_list

def unsupervised_pipeline(pth=None, learner=None, n_clusters=4):
    df = pd.DataFrame()

    if learner is None:
        if pth is None:
            print('No data provided...')
            exit()
        else:
            df = pd.read_csv(pth, index_col=0)
    else:
        df = learner.result_dataset

    kbest_columns = get_drop_colunms(ru_filter=0)

    # Dimensionality Reduction
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df.loc[:, kbest_columns].dropna(axis=1)), index=df.index)

    # Cluster
    km = KMeans(n_clusters=n_clusters)
    df_pca['kmean'] = km.fit_predict(df_pca)
    res_df = pd.concat([df_pca, df], axis=1)

    # Plot Cluster Map
    plotdf = pd.DataFrame(columns=[5, 10, 15, 20, 25], index=['Na', 'Cs', 'Ba', 'K'])
    g = sns.FacetGrid(res_df, col='Ele2', col_wrap=3)
    eles = res_df['Ele2'].unique()
    for i, ele in enumerate(eles):
        for idx, x in res_df.loc[res_df['Ele2'] == ele, ['Ele3', 'Load3', 'kmean']].iterrows():
            plotdf.loc[x.Ele3, x.Load3] = x['kmean']
        plotdf = plotdf.apply(pd.to_numeric)
        sns.heatmap(data=plotdf, cmap=sns.color_palette("hls", n_clusters), ax=g.axes[i], vmin=0,
                    vmax=res_df['kmean'].max(), linewidths=.5, annot=True, cbar=False)
        g.axes[i].set_title(ele)

    plt.savefig('{}\\{}-clustermap.png'.format(skynet.svfl, skynet.svnm), dpi=400)
    plt.close()

    # Plot Scatter of KMeans with groups
    sns.scatterplot(res_df[0], res_df[1], hue=res_df['kmean'], palette=sns.color_palette("hls", n_clusters))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig('{}\\{}-KMeans.png'.format(skynet.svfl, skynet.svnm), dpi=400)
    plt.close()

    # Plot Uncertainty
    plotdf = pd.DataFrame(columns=[5, 10, 15, 20, 25], index=['Na', 'Cs', 'Ba', 'K'])
    g = sns.FacetGrid(res_df, col='Ele2', col_wrap=3)
    eles = res_df['Ele2'].unique()

    for i, ele in enumerate(eles):
        for idx, x in res_df.loc[res_df['Ele2'] == ele, ['Ele3', 'Load3', 'Uncertainty']].iterrows():
            plotdf.loc[x.Ele3, x.Load3] = x.Uncertainty

        plotdf = plotdf.apply(pd.to_numeric)
        sns.heatmap(data=plotdf, cmap='plasma', ax=g.axes[i], vmin=0, vmax=res_df.Uncertainty.max(), linewidths=0.5)
        g.axes[i].set_title(ele)

    plt.savefig('{}\\{}-Uncertainty.png'.format(skynet.svfl, skynet.svnm), dpi=400)


if __name__ == '__main__':
    version = 'v67-bayesian-uncertainty'
    skynet = predict_design_space(version)
    unsupervised_pipeline(learner=skynet, n_clusters=13)