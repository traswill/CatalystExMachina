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


def load_nh3_catalysts(catcont, drop_empty_columns=True):
    """ Import NH3 data from Katie's HiTp dataset(cleaned). """
    df = pd.read_csv(r"..\Data\Processed\AllData_Condensed.csv", index_col=0)

    # Import Cl atoms during synthesis
    cl_atom_df = pd.read_excel(r'..\Data\Catalyst_Synthesis_Parameters.xlsx', index_col=0)

    # Import XRD Peak locations
    xrd_intensity_df = pd.read_csv(r'../Data/Processed/WAXS/WAXS_Peak_Extraction.csv', index_col=0)
    xrd_intensity_lst = np.array(xrd_intensity_df.columns.values, dtype=int).tolist()

    # Import XRD Peak FWHMs
    xrd_fwhm_df = pd.read_csv(r'../Data/Processed/WAXS/WAXS_FWHM_Extraction.csv', index_col=0)
    xrd_fwhm_lst = np.array(xrd_fwhm_df.index.values, dtype=int).tolist()

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

            # cat.feature_add_unsupervised_properties()
            # cat.feature_add_oxidation_states()

            # if row['ID'] in xrd_intensity_lst:
            #     xrd_xs = xrd_intensity_df.index.values
            #     xrd_ys = xrd_intensity_df.loc[:, str(row['ID'])].values
            #     cat.feature_add_xrd_peaks(xrd_xs, xrd_ys)

            # if row['ID'] in xrd_fwhm_lst:
            #     dat = xrd_fwhm_df.loc[row['ID']]
            #     for nm, val in dat.iteritems():
            #         cat.feature_add_xrd_peak_FWHM(peak_nm=nm, peak_fwhm=val)

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


def relearn_with_temps(learner, train_temps, test_temps):
    learner.set_filters(temperature_filter=train_temps)
    learner.filter_master_dataset()
    learner.train_data()
    learner.set_filters(temperature_filter=test_temps)
    learner.filter_master_dataset()


# TODO this is going the way of the dodo
def prediction_pipeline(learner, elements, temps='350orless'):
    def setup(train, test):
        learner.set_filters(temperature_filter=train)
        learner.filter_master_dataset()
        learner.train_data()
        learner.set_filters(temperature_filter=test)
        learner.filter_master_dataset()

    # setup(None, '350orless')
    # learner.predict_from_masterfile(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83], svnm='SS8')
    #
    # setup(None, '350orless')
    # learner.predict_from_masterfile(catids=[84, 85, 86, 87, 89, 90, 91, 93], svnm='SS9')

    setup(None, temps)
    nm = ''.join([x for x in elements])
    learner.predict_all_from_elements(elements=elements, svnm='{}_{}'.format(nm, temps))


def temperature_slice(learner, tslice, kde=False, fold=10):
    for t in tslice:
        learner.set_filters(temperature_filter=t)
        learner.filter_master_dataset()

        learner.train_data()
        learner.extract_important_features(sv=True, prnt=True)
        if fold > 1:
            learner.predict_crossvalidate(kfold=fold, add_to_slave=True)
        elif fold == -1:
            learner.predict_leave_self_out(add_to_slave=True)
        else:
            learner.predict_leave_one_out(add_to_slave=True)
        learner.evaluate_regression_learner()
        learner.preplot_processing()
        learner.save_predictions()
        learner.save_slave()

        g = Graphic(learner=learner)
        g.plot_important_features()
        g.plot_basic()
        g.plot_err()
        g.plot_err(metadata=False, svnm='{}_nometa'.format(learner.svnm))
        if kde:
            normal_feats = ['temperature', 'space_velocity', 'n_Cl_atoms']
            stat_feats = [
                'Second Ionization Energy',
                'Number d-shell Valence Electrons',
                'Dipole Polarizability',
                'Electronegativity',
                'Number Valence Electrons',
                'Conductivity',
                'Covalent Radius',
                'Phi',
                'Heat Fusion',
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


# TODO update this method
def predict_all_binaries():
    SV = 2000
    NH3 = 10
    TMP = 350

    # TODO update method to use catalyst container
    def create_catalyst(e1, w1, e2, w2, e3, w3, tmp, reactnum, space_vel, ammonia_conc):
        cat = CatalystObject()
        cat.ID = 'A'
        cat.add_element(e1, w1)
        cat.add_element(e2, w2)
        cat.add_element(e3, w3)
        cat.input_reactor_number(reactnum)
        cat.input_temperature(tmp)
        cat.input_space_velocity(space_vel)
        cat.input_ammonia_concentration(ammonia_conc)
        cat.feature_add_n_elements()

        feature_generator = {
            0: cat.feature_add_elemental_properties,
            1: cat.add_unweighted_features,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = SupervisedLearner(version='v24-pred')
    catcontainer = CatalystContainer()
    skynet.set_filters(
        element_filter=0,
        # temperature_filter=None,
        # ammonia_filter=1,
        # space_vel_filter=2000,
    )

    model = 'etr'
    skynet.set_learner(learner=model, params=model)

    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + list(range(74, 80)) + [56, 72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    combos = list(itertools.combinations(eles, r=2))

    for vals in combos:
        cat1 = create_catalyst(e1=vals[0], w1=3, e2=vals[1], w2=1, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        catcontainer.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1=vals[0], w1=2, e2=vals[1], w2=2, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        catcontainer.add_catalyst('Predict', cat2)

        cat3 = create_catalyst(e1=vals[0], w1=1, e2=vals[1], w2=3, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        catcontainer.add_catalyst('Predict', cat3)

    load_nh3_catalysts(catcontainer)

    skynet.load_master_dataset(catalyst_container=catcontainer)
    skynet.filter_master_dataset()
    skynet.train_data()
    return skynet, skynet.predict_from_master_dataset()


# TODO update this method
def predict_half_Ru_catalysts():
    SV = 2000
    NH3 = 10
    TMP = 350

    def create_catalyst(e1, w1, e2, w2, e3, w3, tmp, reactnum, space_vel, ammonia_conc):
        cat = CatalystObject()
        cat.ID = 'A'
        cat.add_element(e1, w1)
        cat.add_element(e2, w2)
        cat.add_element(e3, w3)
        cat.input_reactor_number(reactnum)
        cat.input_temperature(tmp)
        cat.input_space_velocity(space_vel)
        cat.input_ammonia_concentration(ammonia_conc)
        cat.feature_add_n_elements()

        feature_generator = {
            0: cat.feature_add_elemental_properties,
            1: cat.add_unweighted_features,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = SupervisedLearner(version='v24-pred')
    catcontainer = CatalystContainer()
    skynet.set_filters(
        element_filter=0,
        # temperature_filter=None,
        # ammonia_filter=1,
        # space_vel_filter=2000,
    )

    model = 'etr'
    skynet.set_learner(learner=model, params=model)

    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + list(range(74, 80)) + [56, 72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values

    for val in eles:
        cat1 = create_catalyst(e1='Ru', w1=0.5, e2=val, w2=4, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        catcontainer.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1='Ru', w1=0.5, e2=val, w2=2, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        catcontainer.add_catalyst('Predict', cat2)

    load_nh3_catalysts(catcontainer)

    skynet.load_master_dataset(catalyst_container=catcontainer)
    skynet.filter_master_dataset()
    skynet.train_data()
    return skynet, skynet.predict_from_master_dataset()


def process_prediction_dataframes(learner, dat_df, svnm='Processed'):
    nm_df = dat_df.loc[:, dat_df.columns.str.contains('Loading')]
    df = pd.DataFrame(dat_df['Predictions'])

    for inx, vals in nm_df.iterrows():
        vals = vals[vals != 0]
        if len(vals) == 2:
            continue
        vals.index = [item[0] for item in vals.index.str.split(' ').values]
        df.loc[inx, 'Element 1'] = vals.index[0]
        df.loc[inx, 'Loading 1'] = vals[0]
        df.loc[inx, 'Element 2'] = vals.index[1]
        df.loc[inx, 'Loading 2'] = vals[1]
        df.loc[inx, 'Element 3'] = vals.index[2]
        df.loc[inx, 'Loading 3'] = vals[2]
        df.loc[inx, 'Name'] = '{}{} {}{} {}{}'.format(vals.index[0], vals[0], vals.index[1], vals[1], vals.index[2], vals[2])

    df.to_csv('{}//{}-{}.csv'.format(learner.svfl, learner.version, svnm))


def unsupervised_first_batch_selection():
    '''
    These prediction bounds were decided upon in July 2018 by Jochen, Katie, Calvin, and myself.
    This code creates the catalyst design space and uses unsupervised ML to determine 64 catalysts for testing.
    '''

    # Initialize Time and unsupervised learning algorithm
    start_time = time.time()
    skynet = Anarchy()

    # Create Element List, elements selected by Katie
    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(44, 51)) + [52] + list(range(55, 59)) + list(range(72, 80)) + [82, 83]

    # Create promoter list (K and Na)
    promoter_list = [11, 19]

    # Create dataframes from these lists
    metal_df = eledf[eledf['Atomic Number'].isin(ele_list)].copy()
    metal_elements = metal_df.index.values
    promoter_df = eledf[eledf['Atomic Number'].isin(promoter_list)].copy()
    promoter_elements = promoter_df.index.values

    # Permute all possible combinations of metals and loadings
    metals = list(itertools.combinations(metal_elements, r=3))
    metal_loadings = list(itertools.permutations([0,1,1,3,3,5], r=3))
    metal_loadings = [tuple(x) for x in set(tuple(x) for x in metal_loadings if np.sum(x) < 8)]

    # Permute all possible combinations of promoters and loadings
    promoters = list(itertools.combinations(promoter_elements, r=2))
    bipromoter_loadings = list(itertools.permutations([0, 5, 10, 10, 15, 20], r=2))
    bipromoter_loadings = [tuple(x) for x in set(tuple(x) for x in bipromoter_loadings)]
    biproms = [(p, pl) for p in promoters for pl in bipromoter_loadings if pl[0] + pl[1] < 21]

    # Use the previous values to permute all possible combinations of metals, promoters, and loadings
    all_combinations = [(m[0], ml[0], m[1], ml[1], m[2], ml[2], p[0][0], p[1][0], p[0][1], p[1][1])
                        for m in metals for ml in metal_loadings for p in biproms]

    # Print resulting informational statistics
    print('Number of Combinations: {}'.format(len(all_combinations)))
    print('Number of metals: {}'.format(len(metals)))
    print('Number of Loadings: {}'.format(len(metal_loadings)))
    print('Number of Promoter Combinations: {}'.format(len(biproms)))
    print('Time to generate samples: {:0.1f} s'.format(time.time() - start_time))

    # Create a dataframe from all combinations, shuffle using .sample()
    catdf = pd.DataFrame(all_combinations, columns=['E1','W1','E2','W2','E3','W3','E4','W4','E5','W5']).sample(
        frac=1,
        random_state=1
    )

    # Print df memory usage
    print("{:03.2f} MB".format(catdf.memory_usage(deep=True).sum() / 1024 ** 2))

    # Create selectors for accessing a subset of the dataframe (mini-batches)
    # This will create 2500 groups of catalysts, and print the number of catalysts per group
    spacing = np.linspace(0, len(catdf), 2500, dtype=int)
    start = spacing[:-1]
    end = spacing[1:]
    print('Number of catalysts per batch: {}'.format(end[0]))

    # Iterate through the batches and run kmedians per batch, save as csv file with batch number
    for i in range(len(start)):
        print('{} out of {}'.format(i, len(start)))
        skynet.index = i
        skynet.set_catalyst_dataframe(catdf.iloc[start[i]:end[i]])
        skynet.build_feature_set()
        skynet.kmeans('..\\Results\\Unsupervised\\Anarchy_NH3_B1\\Anarchy_Batch {}_kmeans_res.csv'.format(i))
        skynet.find_closest_centroid('..\\Results\\Unsupervised\\Anarchy_NH3_B1\\Anarchy_Batch {}_kmedians_res.csv'.format(i))

    print('Time for first batch completion: {:0.1f} s'.format(time.time() - start_time))


def unsupervised_second_batch_selection():
    def compile_previous_batch():
        pths = glob.glob(r'..\Results\Unsupervised\Anarchy_NH3_B1\*kmedians_res.csv')
        compile_df = pd.DataFrame()

        for pth in pths:
            load_df = pd.read_csv(pth, index_col=0)
            compile_df = pd.concat([compile_df, load_df], axis=0)

        return compile_df

    skynet = Anarchy()
    start_time = time.time()
    catdf = compile_previous_batch()
    print('Time to compile batch 1: {:0.1f} s'.format(time.time() - start_time))

    catdf = catdf.sample(frac=1, random_state=0)

    spacing = np.linspace(0, len(catdf), 150, dtype=int)
    start = spacing[:-1]
    end = spacing[1:]
    print('Number of catalysts per batch: {}'.format(end[0]))

    # Iterate through the batches and run kmedians per batch, save as csv file with batch number
    for i in range(len(start)):
        print('{} out of {}'.format(i, len(start)))
        skynet.index = i
        skynet.features = catdf.iloc[start[i]:end[i]]
        skynet.kmeans(sv='..\\Results\\Unsupervised\\Anarchy_NH3_B2\\Anarchy_Batch {}_kmeans_res.csv'.format(i))
        skynet.find_closest_centroid('..\\Results\\Unsupervised\\Anarchy_NH3_B2\\Anarchy_Batch {}_kmedians_res.csv'.format(i))

    print('Time for second batch completion: {:0.1f} s'.format(time.time() - start_time))


def unsupervised_third_batch_selection():
    def compile_previous_batch():
        pths = glob.glob(r'..\Results\Unsupervised\Anarchy_NH3_B2\*kmedians_res.csv')
        compile_df = pd.DataFrame()

        for pth in pths:
            load_df = pd.read_csv(pth, index_col=0)
            load_df = load_df.loc[(load_df['n_elements'] != 0)] # I forgot to exclude all 0 samples, this kills them
            compile_df = pd.concat([compile_df, load_df], axis=0)

        return compile_df

    skynet = Anarchy()
    start_time = time.time()
    catdf = compile_previous_batch()
    print('Time to compile batch 1: {:0.1f} s'.format(time.time() - start_time))
    print('Number of catalysts: {}'.format(len(catdf.index)))

    skynet.features = catdf
    skynet.kmeans(sv=r'..\Results\Unsupervised\Final_kmeans.csv')
    skynet.find_closest_centroid(sv=r'..\Results\Unsupervised\Final_kmedians.csv')

    print('Time for second batch completion: {:0.1f} s'.format(time.time() - start_time))


def extract_final_kmedian():
    df = pd.read_csv(r"..\Results\Unsupervised\Final_kmedians.csv", index_col=0)
    df = df.transpose()[['Loading' in x for x in df.columns]].transpose()

    output_list = list()

    for index, catalyst in df.iterrows():
        catalyst = catalyst[catalyst != 0]
        catalyst.index = [x.replace(' Loading', '') for x in catalyst.index]

        promoters = ''
        metals = ''
        for ele, wt in catalyst.iteritems():
            if ele in ['K', 'Na']:
                promoters += ele
                promoters += str(int(wt*100))
                promoters += ' '
            else:
                metals += ele
                metals += str(int(wt * 100))
                metals += ' '

        cat = metals + promoters
        output_list += [cat]

    df = pd.DataFrame(output_list)
    df.to_csv(r'..\Results\Unsupervised\Anarchy Results.csv')


# Todo update this method
def test_all_ML_models():
    skynet = SupervisedLearner(version='v25')
    catcontainer = CatalystContainer()
    skynet.set_filters(
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        ru_filter=3,
        space_vel_filter=2000
    )

    load_nh3_catalysts(catcontainer)

    # train_elements = ['Ca', 'Mn', 'In']
    train_elements = ['Cu', 'Y', 'Mg', 'Mn', 'Ni', 'Cr', 'W', 'Ca', 'Hf', 'Sc', 'Zn', 'Sr', 'Bi', 'Pd', 'Mo', 'In', 'Rh', 'Ca', 'Mn', 'In']
    df = catcontainer.master_container
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
        element_dataframe = pd.concat([element_dataframe, dat])

    catcontainer.master_container = element_dataframe

    skynet.load_master_dataset(catalyst_container=catcontainer)
    skynet.filter_master_dataset()
    eval_dict = dict()

    for algs in ['rfr','adaboost','tree','neuralnet','svr','knnr','krr','etr','gbr','ridge','lasso']:
        if algs == 'neuralnet':
            skynet.set_learner(learner=algs, params='nnet')
        else:
            skynet.set_learner(learner=algs, params='empty')

        skynet.predict_crossvalidate(kfold=3)
        eval_dict[algs] = mean_absolute_error(skynet.labels_df.values, skynet.predictions)

    print(eval_dict)
    return eval_dict


def plot_all_ML_models(d):
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

    names = d.keys()
    vals = d.values()

    df = pd.DataFrame([names, vals], index=['rgs', 'Mean Absolute Error']).T
    df['Machine Learning Algorithm'] = [nm_dict.get(x, 'ERROR') for x in df['rgs'].values]
    df.sort_values(by='Mean Absolute Error', inplace=True, ascending=False)

    g = sns.barplot(x='Machine Learning Algorithm', y='Mean Absolute Error', data=df, palette="GnBu_d")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.ylim(0,0.36)
    plt.show()


def unsupervised_exploration(learner):
    pth = '{}//{}-unsupervised-slave-dataset.csv'.format(learner.svfl, learner.version)
    df = pd.read_csv(pth, usecols=['Element Dictionary','kmeans'])
    df_new = pd.DataFrame()
    for x in df['Element Dictionary'].values:
        eles = x.split('[')[-1].split(']')[0].replace('\'', '').replace('(','').replace(')','').split(', ')
        df_new = pd.concat([df_new, pd.DataFrame(eles)], axis=1)
    df_new = df_new.transpose()
    df_new.columns = ['Ele1','Wt1','Ele2','Wt2','Ele3','Wt3']
    df_new['group'] = df['kmeans'].values
    df_new.reset_index(drop=True, inplace=True)
    df_new['Wt1'] = df_new['Wt1'].astype(float)
    df_new['Wt2'] = df_new['Wt2'].astype(float)
    df_new['Wt3'] = df_new['Wt3'].astype(float)

    print(df_new)

    # df_new.plot(kind='scatter',x='Wt1',y='Wt2',hue='group')
    sns.swarmplot(x='Wt1', y='Wt2', data=df_new, hue='group')
    plt.show()


def swarmplot_paper1():
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

        # cat = CatalystObject()
        # cat.ID = 'B_{}'.format(atnum)
        # cat.add_element('Ru', 2)
        # cat.add_element(ele, 2)
        # cat.add_element('K', 12)
        # cat.input_group(atnum)
        # cat.feature_add_n_elements()
        # cat.feature_add_Lp_norms()
        # cat.feature_add_elemental_properties()
        # add_obs(cat)
        #
        # catcont.add_catalyst(index=cat.ID, catalyst=cat)
        #
        # cat = CatalystObject()
        # cat.ID = 'C_{}'.format(atnum)
        # cat.add_element('Ru', 1)
        # cat.add_element(ele, 3)
        # cat.add_element('K', 12)
        # cat.input_group(atnum)
        # cat.feature_add_n_elements()
        # cat.feature_add_Lp_norms()
        # cat.feature_add_elemental_properties()
        # add_obs(cat)
        #
        # catcont.add_catalyst(index=cat.ID, catalyst=cat)

    # ***** Set up Catalyst Container to only include specified elements*****
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer, drop_empty_columns=False)

    stupid = False
    if stupid:
        # We had an idea to see how predictions would work using only Ru, RuK, and Al2O3.  This makes that happen.
        pass


    else:
        train_elements = ['Ca', 'Mn', 'Rh', 'W', 'Bi']
        df = catcontainer.master_container
        element_dataframe = pd.DataFrame()

        for ele in train_elements:
            dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
            element_dataframe = pd.concat([element_dataframe, dat])

        catcontainer.master_container = element_dataframe

    # ***** Setup Machine Learning *****
    skynet = SupervisedLearner(version='v42-swarm')
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
    skynet.load_master_dataset(catalyst_container=catcontainer)
    skynet.set_features_to_drop(features=['reactor', 'n_Cl_atoms'])
    skynet.filter_master_dataset()
    skynet.set_training_data()
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
    skynet.load_master_dataset(testcatcontainer)
    skynet.set_features_to_drop(features=['reactor', 'n_Cl_atoms'])
    skynet.drop_features()
    skynet.set_training_data()
    skynet.predict_data()

    # ***** Plot base swarmplot *****
    catdf = testcatcontainer.master_container
    print(skynet.predictions)
    catdf['Predicted'] = skynet.predictions
    # catdf['Predicted'] = [x[1] for x in skynet.predictions]
    df = catdf.loc[:, ['Element Dictionary', 'Predicted', 'temperature']].copy()
    df.reset_index(inplace=True)

    sns.violinplot(x='temperature', y='Predicted', data=df, inner=None, color=".8", scale='count', cut=2.5)
    sns.stripplot(x='temperature', y='Predicted', data=df, jitter=False, linewidth=1)
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Predicted Conversion')

    plt.savefig(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Figures\3Ru_swarmplot_{}.png'.format(''.join(train_elements)))
    plt.close()

    # Save
    catdf.to_csv(r'../Results/3Ru_prediction_data_{}.csv'.format(''.join(train_elements)))
    print(df[df['temperature'] == 300.0].sort_values('Predicted', ascending=False).head())


def unsupervised_paper_1_training_set_selection():
    def create_catalyst(catcont, ele, atnum):
        def add_obs(cat):
            # cat.add_observation(
            #     temperature=250,
            #     space_velocity=2000,
            #     gas_concentration=1,
            #     reactor_number=0
            # )

            cat.add_observation(
                temperature=300,
                space_velocity=2000,
                gas_concentration=1,
                reactor_number=0
            )

            # cat.add_observation(
            #     temperature=350,
            #     space_velocity=2000,
            #     gas_concentration=1,
            #     reactor_number=0
            # )
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

    start_time = time.time()
    skynet = Anarchy()

    # Create Element List, elements selected by Katie
    eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + \
               list(range(45, 51)) + list(range(74, 80)) + [72, 82, 83]

    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    edf = pd.DataFrame([eles, ele_list], index=['Ele', 'Atnum']).T
    eles = edf.values.tolist()

    testcatcontainer = CatalystContainer()

    for nm, atnum in eles:
        create_catalyst(catcont=testcatcontainer, ele=nm, atnum=atnum)

    testcatcontainer.build_master_container(drop_empty_columns=False)
    catdf = testcatcontainer.master_container
    edict = catdf['Element Dictionary'].copy()
    catdf.drop(columns=[
        '- Loading', '-- Loading', 'Element Dictionary', 'group', 'Periodic Table Column_mean',
        'Periodic Table Column_mad', 'Periodic Table Row_mean', 'Periodic Table Row_mad',
        'Mendeleev Number_mean', 'Mendeleev Number_mad'
    ], inplace=True)
    catdf = catdf.loc[:, (catdf != 0).any(axis=0)]

    skynet.set_features(catdf)

    # Test all possible cluster sizes
    # inertia_list = list()
    # for i in range(2, 20):
    #     _, _, inertia = skynet.ammonia_kmeans(clusters=i)
    #     inertia_list += [inertia]
    #     print(inertia)
    #
    # pd.DataFrame(inertia_list).to_csv('..\\Results\\v39-Paper1\\inertia_per_cluster.csv')

    _, _, kmeans = skynet.all_cluster_labels(clusters=5)
    df = pd.DataFrame(edict)
    df['Kmeans'] = kmeans
    df.to_csv('..\\Results\\v39-Paper1\\Anarchy_cluster_results.csv')
    featdf = catdf.copy()
    featdf['Kmeans'] = kmeans
    featdf.to_csv('..\\Results\\v39-Paper1\\Anarchy_features_cluster_results.csv')

    exit()
    skynet.kmeans(clusters=3, sv='..\\Results\\v39-Paper1\\Anarchy_FirstPaperPredictions_kmeans_res.csv')
    res = skynet.find_closest_centroid(
        '..\\Results\\v39-Paper1\\Anarchy_FirstPaperPredictions_kmedians_res.csv')

    res = res.loc[:, ['Loading' in x for x in res.columns]]
    res = res.loc[:, (res != 0).any(axis=0)]
    print(res)

    print('Time for first batch completion: {:0.1f} s'.format(time.time() - start_time))

def categorize_data_from_swarmpredictions():
    # ***** Set up Catalyst Container to only include specified elements*****
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer, drop_empty_columns=False)

    train_elements = ['Ca', 'Mn', 'In']
    df = catcontainer.master_container
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
        element_dataframe = pd.concat([element_dataframe, dat])

    catcontainer.master_container = element_dataframe

    # ***** Setup Machine Learning *****
    skynet = SupervisedLearner(version='v39')
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=None,
        pressure_filter=None
    )

    # ***** Train the learner *****
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_master_dataset(catalyst_container=catcontainer)
    skynet.set_features_to_drop(features=['reactor', 'n_Cl_atoms'])
    skynet.filter_master_dataset()
    skynet.set_training_data()
    skynet.train_data()

    testcatcontainer = CatalystContainer()
    load_nh3_catalysts(testcatcontainer, drop_empty_columns=False)
    skynet.load_master_dataset(testcatcontainer)

    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )
    skynet.set_features_to_drop(features=['reactor', 'n_Cl_atoms'])
    skynet.filter_master_dataset()
    skynet.set_training_data()
    skynet.predict_data()
    skynet.preplot_processing()
    skynet.save_predictions()


def plot_pred_meas_swarm():
    df = pd.read_csv(r"C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v39-Swarmplot-Paper1\3Ru_prediction_data_CaMnIn_inpaper.csv")
    df = df[~df['Metal'].isna()]
    print(df)

    sns.violinplot(x='temperature', y='Predicted', data=df, inner=None, color=".8", scale='count', cut=2.5, width=0.3)
    sns.stripplot(x='temperature', y='Predicted', data=df, jitter=False, linewidth=1)
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Predicted Conversion')
    plt.ylim(0, 1.2)

    plt.savefig(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Figures\CaMnIn_swarmplot_Predicted.png')
    plt.close()

    sns.violinplot(x='temperature', y='Measured', data=df, inner=None, color=".8", scale='count', cut=2.5, width=0.3)
    sns.stripplot(x='temperature', y='Measured', data=df, jitter=False, linewidth=1)
    plt.xlabel('Temperature ($^\circ$C)')
    plt.ylabel('Measured Conversion')
    plt.ylim(0, 1.2)

    plt.savefig(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Figures\CaMnIn_swarmplot_Measured.png')
    plt.close()

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
    skynet.load_master_dataset(catalyst_container=catcontainer)
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


def generate_feature_changes_from_learning_rate():
    pths = glob.glob(r'..\Results\v39-learning-rate\gen_features\*.csv')

    resdf = pd.DataFrame()

    for pth in pths:
        nm = pth.split('\\')[-1].split('.')[0]
        eles = nm.split('-')[0]
        n_eles = sum(1 for x in eles if x.isupper())

        df = pd.read_csv(pth, index_col=0)
        df.sort_values(by='Feature Importance', ascending=False, inplace=True)
        top6 = df.head(6)
        top6.columns = [eles]
        df.columns = [eles]
        resdf = pd.concat([resdf, df.T], sort=True)

    resdf[resdf.isnull()] = 0
    resdf['n_eles'] = [sum(1 for x in elems if x.isupper()) for elems in resdf.index]

    ncatfeatdf = pd.DataFrame()

    for n in range(resdf['n_eles'].min(), resdf['n_eles'].max()):
        res = resdf[resdf['n_eles']==n].sum()
        res = res[res != 0]
        ncatfeatdf = pd.concat([ncatfeatdf, res], axis=1)

    ncatfeatdf.to_csv(r'..\Results\v39-learning-rate\res.csv')


def tune_ert_for_3catalysts():
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer)

    train_elements = ['Ca', 'Mn', 'In']
    df = catcontainer.master_container
    element_dataframe = pd.DataFrame()

    for ele in train_elements:
        dat = df.loc[(df['{} Loading'.format(ele)] > 0) & (df['n_elements'] == 3)]
        element_dataframe = pd.concat([element_dataframe, dat])

    catcontainer.master_container = element_dataframe

    skynet = SupervisedLearner(version='v37-3Ru-CaMnIn')
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=3,
        pressure_filter=None
    )

    skynet.load_master_dataset(catcontainer)
    skynet.filter_master_dataset()
    skynet.set_learner(learner='etr', params='etr')
    skynet.hyperparameter_tuning()


if __name__ == '__main__':
    # ***** Unsupervised Machine Learning Parameter Definition *****
    if False:
        unsupervised_first_batch_selection()
        unsupervised_second_batch_selection()
        unsupervised_third_batch_selection()
        extract_final_kmedian()
        exit()

    # ***** Testing ML Models for Paper *****
    if False:
        d = test_all_ML_models()
        plot_all_ML_models(d)
        exit()

    # ***** Predict all elements from Ca, Mn, In bimetallics (Ru-M-K) *****
    if False:
        # determine_algorithm_learning_rate()
        read_learning_rate(pth=r"C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v44-learning-rate\figures\learning_rate2.csv")
        # generate_feature_changes_from_learning_rate()
        exit()

    if False:
        # unsupervised_paper_1_training_set_selection()
        swarmplot_paper1()
        # categorize_data_from_swarmpredictions()
        # plot_pred_meas_swarm()
        exit()

    # ***** Predict Binaries and 0.5Ru Catalysts within original parameter space *****
    if False:
        skynet, df = predict_all_binaries()
        process_prediction_dataframes(skynet, df, svnm='binaries')

        skynet, df = predict_half_Ru_catalysts()
        process_prediction_dataframes(skynet, df, svnm='half-ru')
        exit()

    # Train with Ru3, predict others
    if False:
        # ***** Set up Catalyst Container*****
        catcontainer = CatalystContainer()
        load_nh3_catalysts(catcont=catcontainer)

        # ***** Begin Machine Learning *****
        skynet = SupervisedLearner(version='v44-Ru2')
        skynet.set_filters(
            element_filter=3,
            temperature_filter='350orless',
            ammonia_filter=1,
            space_vel_filter=2000,
            ru_filter=3,
            pressure_filter=None
        )

        # ***** Load SupervisedLearner *****
        skynet.set_learner(learner='etr', params='etr')
        skynet.load_master_dataset(catalyst_container=catcontainer)
        skynet.set_features_to_drop(features=['reactor'])

        skynet.filter_master_dataset()
        skynet.train_data()
        skynet.extract_important_features(sv=True, prnt=True)

        skynet.set_filters(
            element_filter=3,
            temperature_filter='350orless',
            ammonia_filter=1,
            space_vel_filter=2000,
            ru_filter=2,
            pressure_filter=None
        )
        skynet.filter_master_dataset()

        skynet.predict_data()
        skynet.preplot_processing()
        skynet.save_predictions()

        g = Graphic(learner=skynet)
        g.plot_important_features()
        g.plot_basic()
        g.plot_err()
        g.plot_err(metadata=False, svnm='{}_nometa'.format(skynet.svnm))
        g.bokeh_predictions()

        exit()

    # ***** Set up Catalyst Container*****
    catcontainer = CatalystContainer()
    load_nh3_catalysts(catcont=catcontainer)

    # ***** Begin Machine Learning *****
    skynet = SupervisedLearner(version='v48-only-ndband')
    skynet.set_filters(
        element_filter=3,
        temperature_filter=300,
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=0,
        pressure_filter=None
    )

    # ***** Load SupervisedLearner *****
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_master_dataset(catalyst_container=catcontainer)

    # v45-Without Zpp added
    zpp_list = ['Zunger Pseudopotential (d)', 'Zunger Pseudopotential (p)',
                                          'Zunger Pseudopotential (pi)', 'Zunger Pseudopotential (s)',
                                          'Zunger Pseudopotential (sigma)']

    load_list = ['{} Loading'.format(x) for x in
                 ['Ru','Cu', 'Y', 'Mg', 'Mn',
                  'Ni', 'Cr', 'W', 'Ca', 'Hf',
                  'Sc', 'Zn', 'Sr', 'Bi', 'Pd',
                  'Mo', 'In', 'Rh', 'K']]

    skynet.set_features_to_drop(features=['reactor', 'Periodic Table Column', 'Mendeleev Number'] + zpp_list + load_list)
    skynet.reduce_feature_set()
    skynet.filter_master_dataset()

    # skynet.generate_learning_curve()
    # exit()


    # ***** Tune Hyperparameters *****
    # skynet.filter_master_dataset()
    # skynet.hyperparameter_tuning()
    # exit()

    # ***** General Opreation: temperature_slice method *****
    # fold: k-fold cv if greater than 0, leave-one-out if 0, leave-self-out if -1
    # kde: true or false to generate graphs (false by default to save time)
    #

    # temperature_slice(learner=skynet, tslice=['350orless'], fold=-1) # ['350orless', 250, 300, 350, 400, 450, None]
    temperature_slice(learner=skynet, tslice=['350orless', 250, 300, 350, 400], fold=0, kde=False)

    # relearn_with_temps(learner=skynet, train_temps='350orless', test_temps='350orless')
    # skynet.predict_all_from_elements(elements=['Ca', 'In', 'Mn'], svnm='CaInMn_350orless')
    #
    # relearn_with_temps(learner=skynet, train_temps='350orless', test_temps='350orless')
    # skynet.predict_all_from_elements(elements=['Mg', 'In', 'Mn'], svnm='MgInMn_350orless')
    #
    # relearn_with_temps(learner=skynet, train_temps='350orless', test_temps='350orless')
    # skynet.predict_all_from_elements(elements=['Mg', 'Bi', 'Mn'], svnm='MgBiMn_350orless')
