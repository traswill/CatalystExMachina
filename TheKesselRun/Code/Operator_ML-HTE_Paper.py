# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

from TheKesselRun.Code.LearnerOrder import SupervisedLearner, CatalystContainer
from TheKesselRun.Code.LearnerAnarchy import Anarchy
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation
from TheKesselRun.Code.Plotter import Graphic
from sklearn.feature_selection import SelectKBest, f_regression

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
            cat.set_group(dat['Groups'])
            try:
                cat.add_n_cl_atoms(cl_atom_df.loc[dat['ID'], 'Precursor mol Cl'])
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

def three_catalyst_model(version):
    def create_catalyst(catcont, ele, atnum):
        def add_obs(cat):
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
        # Create a catalyst of 3,1,12 Ru-ele-K for testing
        cat = CatalystObject()
        cat.ID = 'A_{}'.format(atnum)
        cat.add_element('Ru', 3)
        cat.add_element(ele, 1)
        cat.add_element('K', 12)
        cat.set_group(atnum)
        cat.feature_add_n_elements()
        cat.feature_add_Lp_norms()
        cat.feature_add_elemental_properties()
        add_obs(cat)

        catcont.add_catalyst(index=cat.ID, catalyst=cat)

    # ***** Set up Catalyst Container to only include specified elements*****
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer, drop_empty_columns=False)

    train_elements = ['Ca', 'Mn', 'In']
    df = catcontainer.master_container
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
        element_dataframe = pd.concat([element_dataframe, dat])

    # Option for RuK
    # dat = df.loc[(df['Ru Loading'] == 0.04) & (df['n_elements'] == 2)]
    # element_dataframe = pd.concat([element_dataframe, dat])

    catcontainer.master_container = element_dataframe

    # ***** Setup Machine Learning *****
    skynet = SupervisedLearner(version=version)
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    # ***** Train the learner *****
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_static_dataset(catalyst_container=catcontainer)
    skynet.set_target_columns(cols=['Measured Conversion'])
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID'])
    zpp_list = ['Zunger Pseudopotential (d)', 'Zunger Pseudopotential (p)',
                'Zunger Pseudopotential (pi)', 'Zunger Pseudopotential (s)',
                'Zunger Pseudopotential (sigma)']

    skynet.set_drop_columns(cols=['reactor', 'n_Cl_atoms', 'Norskov d-band',
                                  'Periodic Table Column', 'Mendeleev Number'] + zpp_list)

    skynet.filter_static_dataset()
    skynet.train_data()

    # ***** Generate all metals for predictions *****
    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(45, 51)) + list(range(74, 80)) + [72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    # eles = list(zip(eles, ele_list))
    edf = pd.DataFrame([eles, ele_list], index=['Ele','Atnum']).T
    edf = edf[~edf['Ele'].isin(train_elements)]
    eles = edf.values.tolist()

    testcatcontainer = CatalystContainer()

    for nm, atnum in eles:
        create_catalyst(catcont=testcatcontainer, ele=nm, atnum=atnum)

    testcatcontainer.build_master_container(drop_empty_columns=False)
    skynet.load_static_dataset(testcatcontainer)
    skynet.set_training_data()
    skynet.predict_data()

    # ***** Plot base swarmplot *****
    catdf = testcatcontainer.master_container
    catdf['Predicted'] = skynet.predictions
    df = catdf.loc[:, ['Element Dictionary', 'Predicted', 'temperature']].copy()
    df.reset_index(inplace=True)

    sns.violinplot(x='temperature', y='Predicted', data=df, inner=None, color=".8", scale='count', cut=2.5)
    sns.stripplot(x='temperature', y='Predicted', data=df, jitter=False, linewidth=1)
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Predicted Conversion')

    svpth = '{}\\Swarm'.format(skynet.svfl)

    if not os.path.exists(svpth):
        os.makedirs(svpth)

    plt.savefig(r'{}\3Ru_swarmplot_{}.png'.format(svpth, ''.join(train_elements)))
    plt.close()

    catdf.to_csv(r'{}/3Ru_prediction_data_{}.csv'.format(svpth, ''.join(train_elements)))
    print(df[df['temperature'] == 300.0].sort_values('Predicted', ascending=False).head())

def test_ML_models_with_feature_reduction(version, note):
    skynet = load_skynet(version=version, note=note)
    skynet.set_filters(
        element_filter=3,
        temperature_filter=300,
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    # Use the full dataset to calculate the columns to keep, then drop all other features
    skynet.filter_static_dataset()
    feat_selector = SelectKBest(score_func=f_regression, k=20)
    feats = feat_selector.fit_transform(skynet.features, skynet.labels)
    feats = feat_selector.inverse_transform(feats)
    skynet.features_df[:] = feats
    kbest_column_list = list(skynet.features_df.loc[:, skynet.features_df.sum(axis=0) != 0].columns)
    skynet.set_drop_columns(cols=list(set(skynet.features_df.columns) - set(kbest_column_list)))
    skynet.filter_static_dataset()

    # Iterate through all ML algorithms and train a model
    eval_dict = dict()

    for algs in ['rfr','adaboost','tree','neuralnet','svr','knnr','krr','etr','gbr','ridge','lasso']:
        if algs == 'neuralnet':
            skynet.set_learner(learner=algs, params='nnet')
        else:
            skynet.set_learner(learner=algs, params='empty')

        skynet.predict_crossvalidate(kfold=5)
        eval_dict[algs] = mean_absolute_error(skynet.labels_df.values, skynet.predictions)

    print(eval_dict)

    nm_dict = {
        'rfr': 'Random Forest',
        'adaboost': 'AdaBoost',
        'tree': 'Decision Tree',
        'neuralnet': 'Neural Net',
        'svr': 'Support Vector Machine',
        'knnr': 'k-Nearest Neighbor Regression',
        'krr': 'Kernel Ridge Regression',
        'etr': 'Extremely Randomized Trees',
        'gbr': 'Gradient Tree Boosting',
        'ridge': 'Ridge Regressor',
        'lasso': 'Lasso Regressor'
    }

    names = eval_dict.keys()
    vals = eval_dict.values()

    df = pd.DataFrame([names, vals], index=['rgs', 'Mean Absolute Error']).T
    df['Machine Learning Algorithm'] = [nm_dict.get(x, 'ERROR') for x in df['rgs'].values]
    df.sort_values(by='Mean Absolute Error', inplace=True, ascending=False)

    g = sns.barplot(x='Machine Learning Algorithm', y='Mean Absolute Error', data=df, palette="GnBu_d")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.ylim(0,0.36)
    plt.savefig(r'{}\ML_models.png'.format(skynet.svfl))


def test_all_ML_models(version, note, three_ele=True, ru_filter=3):
    skynet = load_skynet(version=version, note=note)
    catcontainer = CatalystContainer()
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=ru_filter,
        pressure_filter=None
    )

    if three_ele:
        train_elements = ['Ca', 'Mn', 'In']
    else:
        train_elements = ['Cu', 'Y', 'Mg', 'Mn', 'Ni', 'Cr', 'W', 'Ca', 'Hf', 'Sc',
                          'Zn', 'Sr', 'Bi', 'Pd', 'Mo', 'In', 'Rh', 'Ca', 'Mn', 'In']

    df = skynet.static_dataset
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['K Loading'] == 0.12) & (df['n_elements'] == 3)]
        element_dataframe = pd.concat([element_dataframe, dat])

    catcontainer.master_container = element_dataframe

    skynet.load_static_dataset(catalyst_container=catcontainer)
    skynet.filter_static_dataset()
    eval_dict = dict()

    for algs in ['rfr','adaboost','tree','neuralnet','svr','knnr','krr','etr','gbr','ridge','lasso']:
        if algs == 'neuralnet':
            skynet.set_learner(learner=algs, params='nnet')
        else:
            skynet.set_learner(learner=algs, params='empty')

        try:
            skynet.predict_crossvalidate(kfold='LOO')
            eval_dict[algs] = mean_absolute_error(skynet.labels_df.values, skynet.predictions)
        except ValueError:
            eval_dict[algs] = -1

    print(eval_dict)

    nm_dict = {
        'rfr': 'Random Forest',
        'adaboost': 'AdaBoost',
        'tree': 'Decision Tree',
        'neuralnet': 'Neural Net',
        'svr': 'Support Vector Machine',
        'knnr': 'k-Nearest Neighbor Regression',
        'krr': 'Kernel Ridge Regression',
        'etr': 'Extremely Randomized Trees',
        'gbr': 'Gradient Tree Boosting',
        'ridge': 'Ridge Regressor',
        'lasso': 'Lasso Regressor'
    }

    names = eval_dict.keys()
    vals = eval_dict.values()

    df = pd.DataFrame([names, vals], index=['rgs', 'Mean Absolute Error']).T
    df['Machine Learning Algorithm'] = [nm_dict.get(x, 'ERROR') for x in df['rgs'].values]
    df.sort_values(by='Mean Absolute Error', inplace=True, ascending=False)

    df.to_csv(r'{}\ML_models.csv'.format(skynet.svfl))

    g = sns.barplot(x='Machine Learning Algorithm', y='Mean Absolute Error', data=df, palette="GnBu_d")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha='right')
    plt.xlabel('Machine learning algorithm')
    plt.ylabel('Mean absolute error')
    plt.tight_layout()
    plt.ylim(0,0.4)
    plt.savefig(r'{}\ML_models.png'.format(skynet.svfl))
    plt.close()

def determine_algorithm_learning_rate(version, note):
    skynet = load_skynet(version=version, note=note)

    elements = [x.replace(' Loading', '') for x in skynet.static_dataset.columns if 'Loading' in x]
    elements.remove('K')
    elements.remove('Ru')

    # To be removed once dataset is complete - temporary in v27 8/8/18
    elements = ['Cu', 'Y', 'Mg', 'Mn', 'Ni', 'Cr', 'W', 'Ca', 'Hf', 'Sc', 'Zn', 'Sr', 'Bi', 'Pd', 'Mo', 'In', 'Rh']
    loads = [0.03, 0.02, 0.01]

    results = pd.DataFrame()
    results_LSOcv = pd.DataFrame()
    results_3foldcv = pd.DataFrame()
    results_10foldcv = pd.DataFrame()

    allcats = [(x, y) for x in elements for y in loads]

    for i in range(2, len(allcats)): # iterate through all possible numbers of catalyst
        print('{} of {}'.format(i, len(allcats)))
        # if i == 3:
        #     for eset in list(itertools.combinations(elements, 3)):
        #         allcats = [(x, y) for x in eset for y in loads]
        #         catalyst_set, load_set = list(zip(*allcats))
        #         df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set,
        #                                               save_plots=False, save_features=False,
        #                                               svnm=''.join(catalyst_set))
        #         mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)
        #
        #         results_3ele.loc['{} {} {}'.format(eset[0], eset[1], eset[2]), 'MAE'] = mae
        #
        #     results_3ele.to_csv(r'../Results/3_element_learning.csv')
        #     results_3ele.to_csv(r'{}/figures/3_element_learning.csv'.format(skynet.svfl))
        #     exit()

        for j in range(1, 25): # randomly sample x catalyst groups
            # eset = random.sample(elements, i)
            # allcats = [(x, y) for x in eset for y in loads]
            catalyst_set, load_set = list(zip(*random.sample(allcats, i)))
            # print('Catalyst Set: {} \n Load Set: {}'.format(catalyst_set, load_set))
            # catalyst_set, load_set = list(zip(*allcats))

            # Predict from training set
            df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set, cv=False,
                                                  save_plots=False, save_features=False,
                                                  svnm=''.join(catalyst_set))
            mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)

            results.loc[i, j] = mae

            # Cross validateb LSO
            df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set, cv='LSO',
                                                  save_plots=False, save_features=False,
                                                  svnm=''.join(catalyst_set))
            mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)

            results_LSOcv.loc[i, j] = mae

            # Cross validateb 3-fold
            try:
                df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set, cv=3,
                                                      save_plots=False, save_features=False,
                                                      svnm=''.join(catalyst_set))
                mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)
            except ValueError:
                mae = np.nan

            results_3foldcv.loc[i, j] = mae

            # Cross validateb 10-fold
            try:
                df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set, cv=10,
                                                      save_plots=False, save_features=False,
                                                      svnm=''.join(catalyst_set))
                mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)
            except ValueError:
                mae = np.nan

            results_10foldcv.loc[i, j] = mae

    # results.to_csv(r'../Results/learning_rate4.csv')
    results.to_csv(r'{}/learning_predict-rate.csv'.format(skynet.svfl))
    results_LSOcv.to_csv(r'{}/learning_LSOcv-rate.csv'.format(skynet.svfl))
    results_3foldcv.to_csv(r'{}/learning_3foldcv-rate.csv'.format(skynet.svfl))
    results_10foldcv.to_csv(r'{}/learning_10foldcv-rate.csv'.format(skynet.svfl))


def read_learning_rate(pth):
    # Prediction Rate
    learn_df = pd.read_csv('{}\\{}'.format(pth, 'learning_predict-rate.csv'), index_col=0)
    datlist = list()
    for idx, rw in learn_df.iterrows():
        for val in rw:
            datlist.append([idx, val])

    learn_df_summary = pd.DataFrame(datlist, columns=['nCatalysts', 'Mean Absolute Error'])

    # LSO CV
    cv_df = pd.read_csv('{}\\{}'.format(pth, 'learning_LSOcv-rate.csv'), index_col=0)
    datlist = list()
    for idx, rw in cv_df.iterrows():
        for val in rw:
            datlist.append([idx, val])

    cv_df_summary = pd.DataFrame(datlist, columns=['nCatalysts', 'Mean Absolute Error'])

    # 3-fold
    cv3_df = pd.read_csv('{}\\{}'.format(pth, 'learning_3foldcv-rate.csv'), index_col=0)
    datlist = list()
    for idx, rw in cv3_df.iterrows():
        for val in rw:
            datlist.append([idx, val])

    cv3_df_summary = pd.DataFrame(datlist, columns=['nCatalysts', 'Mean Absolute Error'])

    # 10-fold
    cv10_df = pd.read_csv('{}\\{}'.format(pth, 'learning_10foldcv-rate.csv'), index_col=0)
    datlist = list()
    for idx, rw in cv10_df.iterrows():
        for val in rw:
            datlist.append([idx, val])

    cv10_df_summary = pd.DataFrame(datlist, columns=['nCatalysts', 'Mean Absolute Error'])

    # plot things
    sns.lineplot(x='nCatalysts', y='Mean Absolute Error', data=learn_df_summary)
    sns.lineplot(x='nCatalysts', y='Mean Absolute Error', data=cv_df_summary)
    sns.lineplot(x='nCatalysts', y='Mean Absolute Error', data=cv3_df_summary)
    sns.lineplot(x='nCatalysts', y='Mean Absolute Error', data=cv10_df_summary)
    plt.legend(['Learning Rate', 'LSO Cross-Validatation', '3-fold Cross-Validatation', '10-fold Cross-Validatation'])
    plt.xlabel('Number of Catalysts in Training Dataset')
    plt.xlim(1, 50)
    plt.yticks(np.arange(0.10, 0.35, 0.05))
    plt.ylim(0.10, 0.25)
    plt.savefig(r'{}//learningrate.png'.format(pth), dpi=400)

def load_skynet(version, note, drop_loads=False, drop_na_columns=True, ru_filter=0, k_filter=True):
    # Load Data
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer, drop_empty_columns=drop_na_columns)

    # Init Learner
    skynet = SupervisedLearner(version=version, note=note)
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
    if k_filter:
        skynet.static_dataset = skynet.static_dataset.loc[skynet.static_dataset['K Loading'] == 0.12]

    # Set parameters
    skynet.set_target_columns(cols=['Measured Conversion'])
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID'])

    if drop_loads:
        load_list = ['{} Loading'.format(x) for x in
                     ['Ru', 'Cu', 'Y', 'Mg', 'Mn',
                      'Ni', 'Cr', 'W', 'Ca', 'Hf',
                      'Sc', 'Zn', 'Sr', 'Bi', 'Pd',
                      'Mo', 'In', 'Rh', 'K']]
    else:
        load_list = []

    skynet.set_drop_columns(
        cols=['reactor', 'Periodic Table Column', 'Mendeleev Number', 'Norskov d-band']
    )

    skynet.filter_static_dataset()
    return skynet


def temperature_slice(learner, tslice, kde=False, fold=10):
    for t in tslice:
        learner.set_filters(temperature_filter=t)
        learner.filter_static_dataset()

        learner.train_data()
        featdf = learner.extract_important_features(sv=True, prnt=True)
        if fold > 1:
            learner.predict_crossvalidate(kfold=fold)
        elif fold == -1:
            learner.predict_crossvalidate(kfold='LSO')
        else:
            learner.predict_crossvalidate(kfold='LOO')

        learner.evaluate_regression_learner()
        learner.compile_results()

        g = Graphic(df=learner.result_dataset, svfl=learner.svfl, svnm=learner.svnm)
        g.plot_important_features(df=featdf)
        g.plot_basic()
        g.plot_err()
        g.plot_err(metadata=False, svnm='{}_nometa'.format(learner.svnm))
        if kde:
            normal_feats = ['temperature', 'space_velocity', 'n_Cl_atoms']
            stat_feats = [
                'Second Ionization Energy',
                'Number d-shell Valence Electrons',
                'Electronegativity',
                'Number Valence Electrons',
                'Conductivity',
                'Covalent Radius',
                'Phi',
                'Polarizability',
                'Melting Temperature',
                'Number d-shell Unfilled Electrons',
                'Number Unfilled Electrons',
                'Fusion Enthalpy'
            ]

            mod_stat_feats = list()
            for x in stat_feats:
                mod_stat_feats += ['{}_mean'.format(x), '{}_mad'.format(x)]

            g.plot_kernel_density(feat_list=normal_feats+mod_stat_feats, margins=False, element=None)

        g.bokeh_predictions()
        # learner.bokeh_by_elements()

def CaMnIn_prediction(version, note):
    # Load the database
    learner = load_skynet(version=version, note=note, ru_filter=0)
    learner.filter_static_dataset()

    # Filter out everything but Ca, Mn, In catalysts, train
    eles = [x.replace(' Loading', '') for x in learner.dynamic_dataset.columns if 'Loading' in x]
    eles = [x for x in eles if x not in ['Ca','Mn','In','Ru','K']]
    learner.filter_out_elements(eles=eles)
    learner.train_data()

    # Reset, filter Ca, Mn, and In out, and predict
    learner.filter_static_dataset()
    learner.filter_out_elements(eles=['Ca','Mn','In'])
    learner.predict_data()

    learner.evaluate_regression_learner()
    learner.compile_results(sv=True)

    g = Graphic(df=learner.result_dataset, svfl=learner.svfl, svnm='{}_CaMnIn'.format(learner.svnm))
    g.plot_basic()
    g.plot_err()
    g.plot_err(metadata=False, svnm='{}_CaMnIn_nometa'.format(learner.svnm))
    plt.close()

def crossvalidation(version, note, ru=0):
    # Load the database
    learner = load_skynet(version=version, note=note, ru_filter=ru)
    learner.filter_static_dataset()
    learner.train_data()

    for cv_task in [3, 10, 'LSO']:
        learner.predict_crossvalidate(kfold=cv_task)
        learner.evaluate_regression_learner()
        learner.compile_results(sv=True, svnm=cv_task)

        g = Graphic(df=learner.result_dataset, svfl=learner.svfl, svnm='{}_{}'.format(learner.svnm, cv_task))
        g.plot_basic()
        g.plot_err()
        g.plot_err(metadata=False, svnm='{}_{}_nometa'.format(learner.svnm, cv_task))
        plt.close()

        g.bokeh_predictions()
        g.bokeh_by_elements()


def element_predictor(elements):
    skynet = load_skynet(version=version, note=note, ru_filter=3, drop_na_columns=False)
    skynet.set_filters(temperature_filter='350orless')
    skynet.filter_static_dataset()
    eles = [x.replace(' Loading', '') for x in skynet.dynamic_dataset.columns if 'Loading' in x]
    eles = [x for x in eles if x not in elements + ['Ru', 'K']]
    skynet.filter_out_elements(eles=eles)
    skynet.train_data()

    catcont = generate_empty_container(ru2=False, ru1=False)
    skynet.load_static_dataset(catcont)
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    nm = ''.join(x for x in elements)
    skynet.compile_results(sv=True, svnm=nm)

    #
    # g = Graphic(df=learner.result_dataset, svfl=learner.svfl, svnm='{}_{}'.format(learner.svnm, nm))
    # g.plot_basic()
    # g.plot_err()
    # g.plot_err(metadata=False, svnm='{}_CaMnIn_nometa'.format(learner.svnm))

def generate_empty_container(ru3=True, ru2=True, ru1=True):
    def create_catalyst(catcont, ele, atnum, ru3, ru2, ru1):
        def add_obs(cat):
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

        # Create a catalyst of 3,1,12 Ru-ele-K for testing
        if ru3:
            cat = CatalystObject()
            cat.ID = 'A_{}'.format(atnum)
            cat.add_element('Ru', 3)
            cat.add_element(ele, 1)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

        # Create a catalyst of 2,2,12 Ru-ele-K for testing
        if ru2:
            cat = CatalystObject()
            cat.ID = 'B_{}'.format(atnum)
            cat.add_element('Ru', 2)
            cat.add_element(ele, 2)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

        # Create a catalyst of 1,3,12 Ru-ele-K for testing
        if ru1:
            cat = CatalystObject()
            cat.ID = 'C_{}'.format(atnum)
            cat.add_element('Ru', 1)
            cat.add_element(ele, 3)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

    # ***** Generate all metals for predictions *****
    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(45, 51)) + list(range(74, 80)) + [72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    edf = pd.DataFrame([eles, ele_list], index=['Ele', 'Atnum']).T
    eles = edf.values.tolist()

    catcont = CatalystContainer()

    for nm, atnum in eles:
        create_catalyst(catcont=catcont, ele=nm, atnum=atnum, ru3=ru3, ru2=ru2, ru1=ru1)

    catcont.build_master_container(drop_empty_columns=False)

    return catcont

def predict_lanthanides(ru3=True, ru2=True, ru1=True):
    def create_catalyst(catcont, ele, atnum, ru3, ru2, ru1):
        def add_obs(cat):
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

        # Create a catalyst of 3,1,12 Ru-ele-K for testing
        if ru3:
            cat = CatalystObject()
            cat.ID = 'A_{}'.format(atnum)
            cat.add_element('Ru', 3)
            cat.add_element(ele, 1)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

        # Create a catalyst of 2,2,12 Ru-ele-K for testing
        if ru2:
            cat = CatalystObject()
            cat.ID = 'B_{}'.format(atnum)
            cat.add_element('Ru', 2)
            cat.add_element(ele, 2)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

        # Create a catalyst of 1,3,12 Ru-ele-K for testing
        if ru1:
            cat = CatalystObject()
            cat.ID = 'C_{}'.format(atnum)
            cat.add_element('Ru', 1)
            cat.add_element(ele, 3)
            cat.add_element('K', 12)
            cat.set_group(atnum)
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()
            add_obs(cat)

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

    def filter(ml, ru):
        ml.set_filters(
            element_filter=3,
            temperature_filter='350orless',
            ammonia_filter=1,
            space_vel_filter=2000,
            ru_filter=ru,
            pressure_filter=None
        )

    def predict(ru3, ru2, ru1, svnm):
        catcont = generate_empty_container(ru3=ru3, ru2=ru2, ru1=ru1)
        skynet.load_static_dataset(catcont)
        skynet.set_training_data()
        skynet.predict_data()
        skynet.compile_results(svnm=svnm)

    def crossvalidate(svnm):
        skynet.predict_crossvalidate(kfold='LOO')
        skynet.compile_results(svnm=svnm)

    def reset_skynet():
        skynet = load_skynet(version=version, note=note, drop_na_columns=False)
        return skynet

    skynet = reset_skynet()
    filter(skynet, ru=0)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()

    # ***** Generate all metals for predictions *****
    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [57, 58, 60, 62, 63, 67, 73]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    edf = pd.DataFrame([eles, ele_list], index=['Ele', 'Atnum']).T
    eles = edf.values.tolist()

    catcont = CatalystContainer()

    for nm, atnum in eles:
        create_catalyst(catcont=catcont, ele=nm, atnum=atnum, ru3=ru3, ru2=ru2, ru1=ru1)

    catcont.build_master_container(drop_empty_columns=False)

    skynet.load_static_dataset(catcont)
    skynet.set_training_data()
    skynet.predict_data()
    skynet.compile_results(svnm='Lanthanides')

def eval_3Ru_vs_3RuplusCaMnIn(version, note):
    skynet = load_skynet(version=version, note=note, drop_na_columns=False)

    # *******************
    # Train 3Ru
    # *******************
    skynet.set_filters(
            element_filter=3,
            temperature_filter='350orless',
            ammonia_filter=1,
            space_vel_filter=2000,
            ru_filter=3,
            pressure_filter=None
        )

    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru')
    g.plot_basic()
    g.plot_err()

    #*******************
    # 3% Ru and 2/1% Ca
    #*******************
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[(df['Ca Loading'] > 0) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&Ca')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru&Ca')
    g.plot_basic()
    g.plot_err()

    # *******************
    # 3% Ru and 2/1% In
    # *******************
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[(df['In Loading'] > 0) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&In')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru&In')
    g.plot_basic()
    g.plot_err()

    # *******************
    # 3% Ru and 2/1% Mn
    # *******************
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[(df['Mn Loading'] > 0) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&Mn')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru&Mn')
    g.plot_basic()
    g.plot_err()

    # *******************
    # 3% Ru and 2/1% CaMn
    # *******************
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0)) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&CaMn')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru&CaMn')
    g.plot_basic()
    g.plot_err()

    # *******************
    # 3% Ru and 2/1% CaMnIn
    # *******************
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&CaMnIn')

    g = Graphic(df=skynet.result_dataset, svfl=skynet.svfl, svnm='3Ru&CaMnIn')
    g.plot_basic()
    g.plot_err()

def predict_all_elements_with_3Ru_21CaMnIn(version, note):
    skynet = load_skynet(version=version, note=note, drop_na_columns=False)

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.filter_static_dataset()
    df = skynet.static_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3) &
            (df['ammonia_concentration'] == 1.0) & (df['temperature'] <= 350)]
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()

    catcont = generate_empty_container(ru3=False)
    skynet.static_dataset = catcont.master_container

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=21,
        pressure_filter=None
    )
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.evaluate_regression_learner()
    skynet.compile_results(svnm='3Ru&CaMnIn2', sv=True)


def make_all_predictions(version, note):
    ''' Generate Crossvalidations and prediction files '''

    def filter(ml, ru):
        ml.set_filters(
            element_filter=3,
            temperature_filter='350orless',
            ammonia_filter=1,
            space_vel_filter=2000,
            ru_filter=ru,
            pressure_filter=None
        )

    def predict(ru3, ru2, ru1, svnm):
        catcont = generate_empty_container(ru3=ru3, ru2=ru2, ru1=ru1)
        skynet.load_static_dataset(catcont)
        skynet.set_training_data()
        skynet.predict_data()
        skynet.compile_results(svnm=svnm, sv=True)

    def crossvalidate(svnm):
        skynet.predict_crossvalidate(kfold='LOO')
        skynet.compile_results(svnm=svnm)

    def reset_skynet(version):
        skynet = load_skynet(version=version, note=note, drop_na_columns=False)
        return skynet

    """ CaMnIn Dataset (3 catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.dynamic_dataset = skynet.dynamic_dataset[
        (skynet.dynamic_dataset['Ca Loading'] > 0) |
        (skynet.dynamic_dataset['Mn Loading'] > 0) |
        (skynet.dynamic_dataset['In Loading'] > 0)
    ]
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='CaMnIn_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.dynamic_dataset = skynet.dynamic_dataset[
        (skynet.dynamic_dataset['Ca Loading'] > 0) |
        (skynet.dynamic_dataset['Mn Loading'] > 0) |
        (skynet.dynamic_dataset['In Loading'] > 0)
        ]
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='CaMnIn')

    """ 3,1,12 RuMK Dataset (17 catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='3Ru_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru')

    """ 3,1,12 RuMK + Full Wt Load CaMnIn Dataset (23 catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    df = skynet.dynamic_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3)]
    skynet.filter_static_dataset()
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='3RuandCaMnInFull_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=3)
    df = skynet.dynamic_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3)]
    skynet.filter_static_dataset()
    skynet.dynamic_dataset = pd.concat([skynet.dynamic_dataset, df])
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=False, ru2=True, ru1=True, svnm='3RuandCaMnInFull')

    """ Full Wt Load CaMnIn Dataset (9 catalysts) """
    skynet = reset_skynet(version)
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3)]
    skynet.dynamic_dataset = df
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='CaMnInFull_CV')

    skynet = reset_skynet(version)
    df = skynet.dynamic_dataset
    df = df[((df['Ca Loading'] > 0) | (df['Mn Loading'] > 0) | (df['In Loading'] > 0)) & (df['n_elements'] == 3)]
    skynet.dynamic_dataset = df
    skynet.dynamic_dataset = skynet.dynamic_dataset[~skynet.dynamic_dataset.index.duplicated(keep='first')]
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=False, ru2=True, ru1=True, svnm='CaMnInFull')

    """ 3,1,12 and 2,2,12 RuMK Dataset (34 Catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=32)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='3Ru_2Ru_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=32)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_2Ru')

    """ 3,1,12 and 2,2,12 RuMK Dataset (34 Catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=31)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='3Ru_1Ru_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=31)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_1Ru')

    """ Full Dataset (51 Catalysts) """
    skynet = reset_skynet(version)
    filter(skynet, ru=0)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    crossvalidate(svnm='3Ru_2Ru_1Ru_CV')

    skynet = reset_skynet(version)
    filter(skynet, ru=0)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_2Ru_1Ru')

def compile_predictions(version):
    """ Begin importing data generated above for compilation into single dataframe """
    pths = glob.glob(
        r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\{}\result*.csv'.format(version)
    )

    output_df = pd.DataFrame(
        columns=['Catalyst', 'Ru Loading', 'Secondary Metal', 'Secondary Metal Loading', 'Temperature',
                 'Measured Conversion', 'CaMnIn Prediction', '3% Ru Prediction', '3% and 2% Ru Prediction',
                 '3% and 1% Ru Prediction', 'All Data Prediction'])

    for pth in pths:
        df = pd.read_csv(pth, index_col=0)
        parse_df = df[['Name', 'Load1', 'Ele2', 'Load2', 'temperature', 'Predicted Conversion']]
        parse_df.columns = ['Catalyst', 'Ru Loading', 'Secondary Metal', 'Secondary Metal Loading', 'Temperature',
                            'Predicted Conversion']
        parse_df.index = ['{}_{}'.format(x[1]['Catalyst'], x[1]['Temperature']) for x in parse_df.iterrows()]

        pdict = {
            '3Ru.csv': '3% Ru Prediction',
            '3Ru_1Ru.csv': '3% and 1% Ru Prediction',
            '3Ru_2Ru.csv': '3% and 2% Ru Prediction',
            '3Ru_2Ru_1Ru.csv': 'All Data Prediction',
            'CaMnIn.csv': 'CaMnIn Prediction'
        }

        col_nm = pdict.get(pth.split('-')[-1], 'Error')

        for idx, val in parse_df.iterrows():
            try:
                output_df.loc[idx, col_nm] = val.values
            except ValueError:
                output_df = pd.concat([output_df, parse_df], sort=False)

    output_df.to_csv(
        r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\{}\compiled_data.csv'.format(version)
    )


def feature_extraction_with_XRD():
    catcont = CatalystContainer()

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
    xrd_df = pd.read_csv(r'..\Data\From Katie\XRD_GrainSize_11Nov18_edited.csv', index_col=0)

    mn = np.floor(xrd_df.index.min())
    mx = np.ceil(xrd_df.index.max())
    idx = np.arange(mn, mx, step=1)
    grain_reindex_df = pd.DataFrame(index=idx, columns=xrd_df.columns)
    for idx, rw in grain_reindex_df.iterrows():
        grain_reindex_df.loc[idx, :] = xrd_df.loc[(xrd_df.index > idx) & (xrd_df.index < idx + 1)].mean().values

    grain_reindex_df = grain_reindex_df.transpose()

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
            cat.set_group(dat['Groups'])
            try:
                cat.add_n_cl_atoms(cl_atom_df.loc[dat['ID']].values[0])
            except KeyError:
                print('Catalyst {} didn\'t have Cl atoms'.format(cat.ID))


            nm = '{:.0f}{:.0f}{:.0f} {}{}{}'.format(dat['Wt1'], dat['Wt2'], dat['Wt3'], dat['Ele1'], dat['Ele2'], dat['Ele3'])
            if '-' not in nm:
                try:
                    xrd = grain_reindex_df.loc[grain_reindex_df.index == nm].fillna(value=0)
                    for twoth, gsz in xrd.iteritems():
                        cat.feature_add(key='{:.0f} 2Th'.format(twoth), value=gsz.values[0])
                except IndexError:
                    print('{} has no XRD'.format(nm))
                    continue
            else:
                continue

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

    catcont.build_master_container(drop_empty_columns=True)

    # Init Learner
    skynet = SupervisedLearner(version=version)
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=0,
        pressure_filter=None
    )

    # Set algorithm and add data
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_static_dataset(catalyst_container=catcont)

    # Set parameters
    skynet.set_target_columns(cols=['Measured Conversion'])
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID'])

    # Lists for dropping certain features
    zpp_list = ['Zunger Pseudopotential (d)', 'Zunger Pseudopotential (p)',
                'Zunger Pseudopotential (pi)', 'Zunger Pseudopotential (s)',
                'Zunger Pseudopotential (sigma)']

    skynet.set_drop_columns(
        cols=['reactor', 'Periodic Table Column', 'Mendeleev Number', 'Norskov d-band', 'n_Cl_atoms']
             + zpp_list + ['Number Unfilled Electrons', 'Number s-shell Unfilled Electrons',
                           'Number p-shell Unfilled Electrons', 'Number d-shell Unfilled Electrons',
                           'Number f-shell Unfilled Electrons'])

    # skynet.set_drop_columns(cols=['reactor', 'Periodic Table Column', 'Mendeleev Number', 'Norskov d-band', 'n_Cl_atoms']
    #                              + zpp_list + load_list)

    skynet.filter_static_dataset()

    skynet.train_data()
    # skynet.predict_crossvalidate(kfold=10)
    skynet.extract_important_features(prnt=True, sv=True)
    skynet.compile_results()
    g = Graphic(df=skynet.result_dataset, svfl=r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v58-XRD')
    g.plot_kernel_density(feat_list=['58 2Th', '28 2Th'], pointcolor='r')
    g.plot_conversion_heatmap(feature1='28 2Th', feature2='Number d-shell Valence Electrons_mad')
    #
    # g.y_axis_value = 'Number d-shell Valence Electrons_mad'
    # g.svnm = '{}_dshell.png'.format(g.svnm.split('.')[0])
    # g.plot_kernel_density(feat_list=['58 2Th', '28 2Th'], pointcolor='r', ylim=None)


if __name__ == '__main__':
    version = 'v83'
    note = 'Changed data files back to nominal 1% conversion'

    CaMnIn_prediction(version, note=note)

    # crossvalidation(version, note, ru=0)

    # test_all_ML_models(version=version, note=note, three_ele=False, ru_filter=0)

    # make_all_predictions(version, note)
    # compile_predictions(version=version)

    # predict_all_elements_with_3Ru_21CaMnIn(version)

    # test_ML_models_with_feature_reduction(version, note)

    # feature_extraction_with_XRD()

    # determine_algorithm_learning_rate(version)

    # read_learning_rate(pth=r"C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v61-learning-rate")

    # eval_3Ru_vs_3RuplusCaMnIn(version, note)

