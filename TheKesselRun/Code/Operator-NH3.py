from TheKesselRun.Code.LearnerOrder import Learner, CatalystContainer
from TheKesselRun.Code.LearnerAnarchy import Anarchy
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation
from TheKesselRun.Code.Plotter import Graphic

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import time
import scipy.cluster
import glob


def load_nh3_catalysts(catcont):
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
                activity=dat['Concentration'],
                selectivity=None
            )
        else:
            cat = CatalystObject()
            cat.ID = dat['ID']
            cat.add_element(dat['Ele1'], dat['Wt1'])
            cat.add_element(dat['Ele2'], dat['Wt2'])
            cat.add_element(dat['Ele3'], dat['Wt3'])
            cat.input_group(dat['Groups'])
            cat.input_n_Cl_atoms(cl_atom_df.loc[dat['ID']].values[0])
            cat.input_standard_error(dat['Standard Error'])
            cat.input_n_averaged_samples(dat['nAveraged'])
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties()

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
                activity=dat['Concentration'],
                selectivity=None
            )

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

    catcont.build_master_container()

def prediction_pipeline(learner):
    # def predict_catalysts(eles, svnm):
    #     learner.set_temp_filter(None)
    #     learner.filter_master_dataset()
    #     learner.train_data()
    #     learner.set_temp_filter('350orless')
    #     learner.filter_master_dataset()
    #     learner.parse_element_dictionary()
    #     exit()
    #
    #     catlst = [1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 43, 44, 45, 49, 50, 51, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67,
    #               68, 69, 73, 74, 75, 76, 77, 78, 85, 86, 87, 89, 90, 106, 107, 108]
    #     learner.predict_from_masterfile(catids=catlst, svnm='CuMgMnPdReRh')
    #
    # predict_catalysts(eles=['Cu','Mg','Mn','Pd','Re','Rh'], svnm='CuMgMnPdReRh')

    def setup():
        learner.set_temp_filter(None)
        learner.filter_master_dataset()
        learner.train_data()
        learner.set_temp_filter('350orless')
        learner.filter_master_dataset()

    setup()
    learner.predict_from_masterfile(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83], svnm='SS8')

    setup()
    learner.predict_from_masterfile(catids=[84, 85, 86, 87, 89, 90, 91, 93], svnm='SS9')

    # Predict all from Cu, Mg, Mn, Pd, Re, Rh
    learner.set_temp_filter(None)
    learner.filter_master_dataset()
    learner.train_data()
    learner.set_temp_filter('350orless')
    learner.filter_master_dataset()

    CuMgMnPdReRh = [1,2,3,4,5,6,29,30,31,32,43,44,45,49,50,51,55,56,57,58,59,60,61,62,63,64,67,68,69,73,74,75,76,77,
                    78,85,86,87,89,90,106,107,108]
    setup()
    learner.predict_from_masterfile(catids=CuMgMnPdReRh, svnm='CuMgMnPdReRh')

    NiPdIrPt = [1,2,3,4,5,6,15,16,17,25,27,28,29,30,31,32,33,34,35,36,37,38,39,43,44,45,49,50,51,55,56,57,58,59,60,
                61,62,63,64,65,66,67,68,69,73,74,75,76,77,78,85,86,87,91,93,106,107,108,112,114,115,116,117,118,119,
                121,123,124,125,126]
    setup()
    learner.predict_from_masterfile(catids=NiPdIrPt, svnm='NiPdIrPt')

    NiPdIrPtCu = [1,2,3,4,5,6,25,27,28,29,30,31,32,33,34,35,36,37,38,39,43,44,45,49,50,51,55,56,57,58,59,60,61,62,
                  63,64,65,66,67,68,69,73,74,75,76,77,78,85,86,87,91,93,106,107,108,112,114,115,116,117,118,119,121,
                  123,124,125,126]
    setup()
    learner.predict_from_masterfile(catids=NiPdIrPtCu, svnm='NiPdIrPtCu')

    HfYScCaSrMg = [1,2,3,4,5,6,15,16,17,25,27,28,36,37,38,39,40,41,42,43,44,45,49,50,51,65,66,67,68,69,76,77,78,82,
                   83,84,85,86,87,89,90,91,93,106,107,108,112,113,114,115,116,117,118,119,120,122,124,125]
    setup()
    learner.predict_from_masterfile(catids=HfYScCaSrMg, svnm='HfYScCaSrMg')

def temperature_slice(learner, tslice):
    for t in tslice:
        learner.set_temp_filter(t)
        learner.filter_master_dataset()

        learner.train_data()
        learner.extract_important_features(sv=True, prnt=True)
        learner.predict_crossvalidate(kfold=10)
        if learner.regression:
            learner.evaluate_regression_learner()
        else:
            learner.evaluate_classification_learner()
        learner.preplot_processing()
        learner.save_predictions()
        g = Graphic(learner=learner)
        g.plot_important_features()
        g.plot_basic()
        g.plot_err()
        g.plot_err(metadata=False, svnm='{}_nometa'.format(learner.svnm))
        g.plot_kernel_density(feat_list=['temperature',
                                         'Ru Loading',
                                         'Rh Loading',
                                         'Second Ionization Energy_wt-mad',
                                         'Second Ionization Energy_wt-mean',
                                         'Number d-shell Valence Electrons_wt-mean',
                                         'Number d-shell Valence Electrons_wt-mad',
                                         'Periodic Table Column_wt-mean',
                                         'Periodic Table Column_wt-mad',
                                         'Electronegativity_wt-mean',
                                         'Electronegativity_wt-mad',
                                         'Number Valence Electrons_wt-mean',
                                         'Number Valence Electrons_wt-mad',
                                         'Conductivity_wt-mean',
                                         'Conductivity_wt-mad',
                                         ], margins=False, element=None)

        # g.plot_err(color_bounds=(250, 450))
        # g.plot_err(metadata=False, svnm='{}_nometa'.format(learner.svnm), color_bounds=(250, 450))

        # Re-add these html generators once moved to Graphic
        g.bokeh_predictions()
        # learner.bokeh_by_elements()


def unsupervised_pipline(learner):
    learner.filter_master_dataset()
    learner.unsupervised_data_segmentation(n_clusters=2)
    learner.set_learner(learner='etr', params='etr')
    learner.train_data()
    learner.predict_crossvalidate(kfold=3)
    learner.evaluate_regression_learner()
    learner.preplot_processing()
    learner.plot_error(metadata=True)


def predict_all_binaries():
    SV = 2000
    NH3 = 10
    TMP = 350

    # TODO migrate into learner
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
            1: cat.feature_add_statistics,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = Learner(
        average_data=True,
        element_filter=0,
        # temperature_filter=None,
        # ammonia_filter=1,
        # space_vel_filter=2000,
        version='v24-pred',
        regression=True
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
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1=vals[0], w1=2, e2=vals[1], w2=2, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        skynet.add_catalyst('Predict', cat2)

        cat3 = create_catalyst(e1=vals[0], w1=1, e2=vals[1], w2=3, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        skynet.add_catalyst('Predict', cat3)

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    skynet.train_data()
    return skynet, skynet.predict_dataset()


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
            1: cat.feature_add_statistics,
            2: cat.feature_add_weighted_average
        }
        feature_generator.get(0, lambda: print('No Feature Generator Selected'))()

        return cat

    skynet = Learner(
        average_data=True,
        element_filter=0,
        # temperature_filter=None,
        # ammonia_filter=1,
        # space_vel_filter=2000,
        version='v24-pred',
        regression=True
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
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1='Ru', w1=0.5, e2=val, w2=2, e3='K', w3=12,
                               tmp=TMP, reactnum=1, space_vel=SV, ammonia_conc=NH3)
        skynet.add_catalyst('Predict', cat2)

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    skynet.train_data()
    return skynet, skynet.predict_dataset()

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
        random_state=0
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


def test_all_ML_models():
    from sklearn.metrics import r2_score, explained_variance_score, \
        mean_absolute_error, roc_curve, recall_score, precision_score, mean_squared_error

    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        space_vel_filter=2000,
        version='v25',
        regression=True
    )

    load_nh3_catalysts(skynet, featgen=0)
    skynet.filter_master_dataset()
    eval_dict = dict()

    for algs in ['rfr','adaboost','tree','neuralnet','svr','knnr','krr','etr','gbr','ridge','lasso']:
        if algs == 'neuralnet':
            skynet.set_learner(learner=algs, params='nnet')
        else:
            skynet.set_learner(learner=algs, params='empty')

        skynet.predict_crossvalidate(kfold=10)
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

    # names = ['Random Forest', 'Adaboost', 'Decision Tree', 'Neural Net', 'Support Vector Machine',
    #          'k-Nearest Neighbor Regression', 'Kernel Ridge Regression', 'Extra Tree Regressor',
    #          'Gradient Boosting Regressor', 'Ridge Regressor', 'Lasso Regressor']
    # vals = [0.121, 0.158, 0.152, 0.327, 0.327, 0.245, 0.168, 0.109, 0.119, 0.170, 0.188]
    # df = pd.DataFrame([names, vals],  index=['Algorithm', 'Mean Absolute Error']).T

    names = d.keys()
    vals = d.values()

    df = pd.DataFrame([names, vals], index=['rgs', 'Mean Absolute Error']).T
    df['Machine Learning Algorithm'] = [nm_dict.get(x, 'ERROR') for x in df['rgs'].values]
    df.sort_values(by='Mean Absolute Error', inplace=True, ascending=False)

    g = sns.barplot(x='Machine Learning Algorithm', y='Mean Absolute Error', data=df, palette="GnBu_d")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
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




if __name__ == '__main__':
    # unsupervised_first_batch_selection()
    # unsupervised_second_batch_selection()
    # unsupervised_third_batch_selection()
    # extract_final_kmedian()
    # exit()

    # generate_kde_plots(feature='Second Ionization Energy_wt-mad')
    # exit()

    # ***** Testing ML Models for Paper *****
    # d = test_all_ML_models()
    # plot_all_ML_models(d)
    # exit()

    # ***** Predict Binaries and 0.5Ru Catalysts *****
    # skynet, df = predict_all_binaries()
    # process_prediction_dataframes(skynet, df, svnm='binaries')
    #
    # skynet, df = predict_half_Ru_catalysts()
    # process_prediction_dataframes(skynet, df, svnm='half-ru')
    # exit()

    # ***** *****
    catcont = CatalystContainer()
    load_nh3_catalysts(catcont=catcont)

    # ***** Begin Machine Learning *****
    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        ammonia_filter=1,
        space_vel_filter=None,
        ru_filter=None,
        pressure_filter=None,
        version='v35-predict-all-from-few',
        regression=True
    )

    # load_nh3_catalysts(learner=skynet, featgen=0)  # 0 is elemental, 1 is statistics,  2 is statmech

    # ***** Unsupervised Learning
    # unsupervised_pipline(skynet)
    # unsupervised_exploration(skynet)
    # exit()

    # ***** Load Learner *****
    # Options: rfr, etr, gbr, rfc (not operational)
    model = 'etr'
    skynet.set_learner(learner=model, params=model)

    # ***** Tune Hyperparameters *****
    # skynet.filter_master_dataset()
    # skynet.hyperparameter_tuning()
    # exit()

    # ***** General Opreation *****
    # temperature_slice(learner=skynet, tslice=['350orless', 250, 300, 350]) # ['350orless', 250, 300, 350, 400, 450, None]
    prediction_pipeline(learner=skynet)