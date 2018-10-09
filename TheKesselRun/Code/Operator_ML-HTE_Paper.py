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
            cat.feature_add_Norskov_dband()
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
        cat.input_group(atnum)
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

def test_all_ML_models(version, three_ele=True, ru_filter=3):
    skynet = SupervisedLearner(version=version)
    catcontainer = CatalystContainer()
    skynet.set_filters(
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        ru_filter=ru_filter,
        space_vel_filter=2000
    )

    load_nh3_catalysts(catcontainer)

    if three_ele:
        train_elements = ['Ca', 'Mn', 'In']
    else:
        train_elements = ['Cu', 'Y', 'Mg', 'Mn', 'Ni', 'Cr', 'W', 'Ca', 'Hf', 'Sc',
                          'Zn', 'Sr', 'Bi', 'Pd', 'Mo', 'In', 'Rh', 'Ca', 'Mn', 'In']

    df = catcontainer.master_container
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
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

        skynet.predict_crossvalidate(kfold=3)
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

def determine_algorithm_learning_rate():
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer)
    elements = [x.replace(' Loading', '') for x in catcontainer.master_container.columns if 'Loading' in x]
    elements.remove('K')
    elements.remove('Ru')

    # To be removed once dataset is complete - temporary in v27 8/8/18
    elements = ['Cu', 'Y', 'Mg', 'Mn', 'Ni', 'Cr', 'W', 'Ca', 'Hf', 'Sc', 'Zn', 'Sr', 'Bi', 'Pd', 'Mo', 'In', 'Rh']
    loads = [0.03, 0.02, 0.01]

    skynet = SupervisedLearner(version='v44-learning-rate')
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=0,
        pressure_filter=None
    )

    skynet.set_learner(learner='etr', params='etr')
    skynet.load_static_dataset(catalyst_container=catcontainer)
    skynet.set_features_to_drop(features=['reactor'])

    results = pd.DataFrame()

    allcats = [(x, y) for x in elements for y in loads]

    for i in range(1, len(allcats)): # iterate through all possible numbers of catalyst
        for j in range(1, 25): # randomly sample x catalyst groups
            catalyst_set, load_set = list(zip(*random.sample(allcats, i)))
            df = skynet.predict_all_from_elements(elements=catalyst_set, loads=load_set,
                                                  save_plots=False, save_features=False,
                                                  svnm=''.join(catalyst_set))
            mae = mean_absolute_error(df['Measured Conversion'].values, df['Predicted Conversion'].values)

            results.loc[i, j] = mae

    results.to_csv(r'../Results/learning_rate2.csv')
    results.to_csv(r'{}/figures/learning_rate2.csv'.format(skynet.svfl))


def read_learning_rate(pth):
    df = pd.read_csv(pth, index_col=0)
    datlist = list()
    for idx, rw in df.iterrows():
        for val in rw:
            datlist.append([idx, val])

    df2 = pd.DataFrame(datlist, columns=['nCatalysts', 'Mean Absolute Error'])
    sns.lineplot(x='nCatalysts', y='Mean Absolute Error', data=df2)
    plt.xlabel('Number of Catalysts in Training Dataset')
    plt.xlim(1, 40)
    plt.yticks(np.arange(0.1, 0.35, 0.05))
    plt.ylim(0.1, 0.3)

    plt.savefig(r'../Figures/ERT_learning_rate5.png', dpi=400)

def load_skynet(version, drop_loads=False, drop_na_columns=True):
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
        ru_filter=3,
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

    skynet.set_drop_columns(cols=['reactor', 'Periodic Table Column', 'Mendeleev Number'] + zpp_list + load_list)
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

        g = Graphic(learner=learner, df=learner.result_dataset)
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
            cat.input_group(atnum)
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
            cat.input_group(atnum)
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
            cat.input_group(atnum)
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


def make_all_predictions(version):
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

    skynet = load_skynet(version=version, drop_na_columns=False)
    skynet.set_target_columns(cols=['Measured Conversion'])
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID'])
    zpp_list = ['Zunger Pseudopotential (d)', 'Zunger Pseudopotential (p)',
                'Zunger Pseudopotential (pi)', 'Zunger Pseudopotential (s)',
                'Zunger Pseudopotential (sigma)']
    skynet.set_drop_columns(cols=['reactor', 'n_Cl_atoms', 'Norskov d-band',
                                  'Periodic Table Column', 'Mendeleev Number'] + zpp_list)

    """ CaMnIn Dataset """
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.dynamic_dataset = skynet.dynamic_dataset[
        (skynet.dynamic_dataset['Ca Loading'] > 0) |
        (skynet.dynamic_dataset['Mn Loading'] > 0) |
        (skynet.dynamic_dataset['In Loading'] > 0)
    ]
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=False, ru1=False, svnm='CaMnIm')

    """ 3,1,12 RuMK Dataset (17 catalysts) """
    filter(skynet, ru=3)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru')

    """ 3,1,12 and 2,2,12 RuMK Dataset (34 Catalysts) """
    filter(skynet, ru=32)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_2Ru')

    """ 3,1,12 and 2,2,12 RuMK Dataset (34 Catalysts) """
    filter(skynet, ru=31)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_1Ru')

    """ Full Dataset (51 Catalysts) """
    filter(skynet, ru=0)
    skynet.filter_static_dataset()
    skynet.set_training_data()
    skynet.train_data()
    predict(ru3=True, ru2=True, ru1=True, svnm='3Ru_2Ru_1Ru')

def compile_predictions(version):
    """ Begin importing data generated above for compilation into single dataframe """
    pths = glob.glob(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v52\result*.csv')
    output_df = pd.DataFrame(
        columns=['Catalyst', 'Ru Loading', 'Secondary Metal', 'Secondary Metal Loading', 'Temperature',
                 'Measured Conversion', 'CaMnIn Prediction', '3% Ru Prediction', '3% and 2% Ru Prediction',
                 '3% and 1% Ru Prediction', 'All Data Prediction'])



    for pth in pths:
        df = pd.read_csv(pth, index_col=0)
        print(pth)
        parse_df = df[['Name', 'Load1', 'Ele2', 'Load2', 'temperature', 'Predicted Conversion']]
        parse_df.columns = ['Catalyst', 'Ru Loading', 'Secondary Metal', 'Secondary Metal Loading', 'Temperature',
                            'Predicted Conversion']
        parse_df.index = ['{}_{}'.format(x[1]['Catalyst'], x[1]['Temperature']) for x in parse_df.iterrows()]

        pdict = {
            '3Ru': '3% Ru Prediction',
            '3Ru_1Ru': '3% and 1% Ru Prediction',
            '3Ru_2Ru': '3% and 2% Ru Prediction',
            '3Ru_2Ru_1Ru': 'All Data Prediction',
            'CaMnIm': 'CaMnIn Prediction'
        }

        col_nm = pdict.get(pth.split('-')[-1], 'Error')

        for idx, val in parse_df.iterrows():
            print(idx)
            try:
                output_df.loc[idx, col_nm] = val.values
            except ValueError:
                output_df = pd.concat([output_df, parse_df], sort=False)

    output_df.to_csv(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v52\compiled_data.csv')



if __name__ == '__main__':
    version = 'v53'
    # make_all_predictions(version=version)
    compile_predictions(version=version)

    # skynet = load_skynet()
    # temperature_slice(learner=skynet, tslice=['350orless', 250, 300, 350], fold=0, kde=False)

    # three_catalyst_model()
    # test_all_ML_models()
