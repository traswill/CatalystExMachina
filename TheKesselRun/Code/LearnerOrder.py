# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

from TheKesselRun.Code.Plotter import Graphic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, BoundaryNorm, to_hex, Normalize
from matplotlib.cm import get_cmap

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, \
    ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict, GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, explained_variance_score, \
    mean_absolute_error, roc_curve, recall_score, precision_score, mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, FeatureAgglomeration
from sklearn.feature_selection import VarianceThreshold, univariate_selection

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker, CustomJS, Slider, Select
from bokeh.plotting import figure, show, output_file, save, curdoc
import bokeh.palettes as pals
from bokeh.models import Range1d, DataRange1d
from bokeh.layouts import row, widgetbox, column, layout

import seaborn as sns
import os
import time

from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation


class CatalystContainer(object):
    def __init__(self):
        self.catalyst_dictionary = dict()
        self.master_container = pd.DataFrame()

    def add_catalyst(self, index, catalyst):
        """ Check if CatalystObject() exists in self.catalyst_dictionary.  Add if not.  Append if it does. """
        if index in self.catalyst_dictionary:
            for key, obs in self.catalyst_dictionary[index].observation_dict:
                self.catalyst_dictionary[index].add_observation(
                    temperature=obs.temerature,
                    space_velocity=obs.space_velocity,
                    gas=obs.gas,
                    gas_concentration=obs.gas_concentration,
                    pressure=obs.pressure,
                    reactor_number=obs.reactor_number,
                    activity=obs.activity,
                    selectivity=obs.selectivity
                )
        else:
            self.catalyst_dictionary[index] = catalyst

    def build_master_container(self):
        # Set up catalyst loading dictionary with loadings
        loading_df = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation'], index_col='Abbreviation').transpose()
        loading_df.columns = ['{} Loading'.format(ele) for ele in loading_df.columns]

        for catid, catobj in self.catalyst_dictionary.items():
            # Reset loading dictionary
            load_df = loading_df.copy()

            # Add elements and loading to loading dict
            for ele, wt in catobj.elements.items():
                load_df.loc[catid, '{} Loading'.format(ele)] = wt / 100

            # Create DF from inputs
            inputdf = pd.DataFrame.from_dict(catobj.input_dict, orient='index').transpose()
            inputdf.index = [catid]

            # Create DF from features
            featdf = pd.DataFrame.from_dict(catobj.feature_dict, orient='index').transpose()
            featdf.index = [catid]

            # Create DF from activity
            actdf = pd.DataFrame(catobj.activity, index=[catid], columns=['Measured Conversion'])

            # Create element dictionary
            eldictdf = pd.DataFrame(catobj.elements.items(), index=[catid], columns=['Element Dictionary'])

            # Combine DFs
            df = pd.concat([load_df, inputdf, featdf, actdf, eldictdf], axis=1)
            self.master_container = pd.concat([self.master_container, df], axis=0)

        self.master_container.dropna(how='all', axis=1, inplace=True)
        self.master_container.fillna(value=0, inplace=True)


class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self, element_filter=3, temperature_filter=None, ammonia_filter=None, space_vel_filter=None,
                 ru_filter=None, pressure_filter=None,
                 version='v00', average_data=True, regression=True):
        """ Put Words Here """

        '''Initialize dictionary to hold import data'''
        self.catalyst_dictionary = dict()

        '''Initialize DataFrames for unchanging data (master) and sorting/filtering (slave)'''
        self.master_dataset = pd.DataFrame()
        self.slave_dataset = pd.DataFrame()
        self.test_dataset = pd.DataFrame()

        '''Initialize sub-functions from the slave dataset.'''
        self.features_df = pd.DataFrame()
        self.labels_df = pd.DataFrame()
        self.plot_df = pd.DataFrame()
        self.predictions = list()
        self.feature_importance_df = pd.DataFrame()

        '''Initialize ML algorithm and tuning parameters'''
        self.machina = None

        self.clustered_machina_dictionary = dict()
        self.n_clusters = None
        self.cluster = None
        self.groups = None

        '''Initialize all options for the algorithm.  These are used in naming files.'''
        self.average_data = average_data
        self.element_filter = element_filter
        self.temperature_filter = temperature_filter
        self.ammonia_filter = ammonia_filter
        self.ru_filter = ru_filter
        self.pressure_filter = pressure_filter
        self.sv_filter = space_vel_filter
        self.group_style = 'full-blind'
        self.version = version
        self.regression = regression

        '''Create path, folder, and filename'''
        self.svfl = '..//Results//{version}_{type}'.format(version=version, type='r' if regression else 'c')
        self.svnm = '{nm}_{type}-{nele}-{temp}'.format(
            nm=version,
            type = 'r' if regression else 'c',
            nele=element_filter,
            temp='{}C'.format(temperature_filter) if temperature_filter is not None else 'All'
        )

        if not os.path.exists(self.svfl):
            os.makedirs(self.svfl)
            os.makedirs('{}\\{}'.format(self.svfl, 'trees'))
            os.makedirs('{}\\{}'.format(self.svfl, 'figures'))
            os.makedirs('{}\\{}'.format(self.svfl, 'htmls'))
            os.makedirs('{}\\{}'.format(self.svfl, 'features'))
            os.makedirs('{}\\{}'.format(self.svfl, 'eval'))

        '''Initialize Time for run-length statistics'''
        self.start_time = time.time()

    def set_name_paths(self):
        self.svfl = '..//Results//{version}_{type}'.format(version=self.version, type='r' if self.regression else 'c')
        self.svnm = '{nm}_{type}-{nele}-{temp}'.format(
            nm=self.version,
            type='r' if self.regression else 'c',
            nele=self.element_filter,
            temp='{}C'.format(self.temperature_filter) if self.temperature_filter is not None else 'All'
        )

    def set_temp_filter(self, temp_filter):
        """ Set Temperature Filter to allow multiple slices without reloading the data """
        self.temperature_filter = temp_filter
        self.set_name_paths()

    def set_ammonia_filter(self, ammonia_filter):
        pass

    def set_space_vel_filter(self, space_vel):
        pass

    def save_filter_config(self):
        pass

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

    def create_master_dataset(self):
        # Set up catalyst loading dictionary with loadings
        loading_df = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation'], index_col='Abbreviation').transpose()
        loading_df.columns = ['{} Loading'.format(ele) for ele in loading_df.columns]

        for catid, catobj in self.catalyst_dictionary.items():
            # Reset loading dictionary
            load_df = loading_df.copy()

            # Add elements and loading to loading dict
            for ele, wt in catobj.elements.items():
                load_df.loc[catid, '{} Loading'.format(ele)] = wt / 100

            # Create DF from inputs
            inputdf = pd.DataFrame.from_dict(catobj.input_dict, orient='index').transpose()
            inputdf.index = [catid]

            # Create DF from features
            featdf = pd.DataFrame.from_dict(catobj.feature_dict, orient='index').transpose()
            featdf.index = [catid]

            # Create DF from activity
            actdf = pd.DataFrame(catobj.activity, index=[catid], columns=['Measured Conversion'])

            # Create element dictionary
            eldictdf = pd.DataFrame(catobj.elements.items(), index=[catid], columns=['Element Dictionary'])

            # Combine DFs
            df = pd.concat([load_df, inputdf, featdf, actdf, eldictdf], axis=1)
            self.master_dataset = pd.concat([self.master_dataset, df], axis=0)

        self.master_dataset.dropna(how='all', axis=1, inplace=True)
        self.master_dataset.fillna(value=0, inplace=True)

    def filter_master_dataset(self):
        """ Filters data from import file for partitioned model training """
        filt = None

        def join_all_indecies(ind_list):
            start_filter = ind_list.pop(0)
            for ind_obj in ind_list:
                start_filter = list(set(start_filter) & set(ind_obj))
            return start_filter

        def filter_elements(ele_filter):
            filter_dict_neles = {
                1: self.master_dataset[self.master_dataset['n_elements'] == 1].index,
                2: self.master_dataset[self.master_dataset['n_elements'] == 2].index,
                3: self.master_dataset[self.master_dataset['n_elements'] == 3].index,
                23: self.master_dataset[(self.master_dataset['n_elements'] == 2) |
                                        (self.master_dataset['n_elements'] == 3)].index,
            }

            if self.element_filter is list():
                n_ele_slice = filter_dict_neles.get(ele_filter, self.master_dataset.index)
            else:
                n_ele_slice = filter_dict_neles.get(ele_filter, self.master_dataset.index)

            return n_ele_slice

        def filter_temperature(temp_filter):
            if self.temperature_filter is None:
                temp_slice = self.master_dataset[self.master_dataset.loc[:, 'temperature'] != 150].index
            elif isinstance(self.temperature_filter, str):
                temp_dict = {
                    'not450': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 450) &
                                                  (self.master_dataset.loc[:, 'temperature'] != 150)].index,
                    'not400': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 400) &
                                                  (self.master_dataset.loc[:, 'temperature'] != 150)].index,
                    'not350': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 350) &
                                                  (self.master_dataset.loc[:, 'temperature'] != 150)].index,
                    '350orless': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 450) &
                                                     (self.master_dataset.loc[:, 'temperature'] != 400) &
                                                     (self.master_dataset.loc[:, 'temperature'] != 150)].index,
                    '300orless': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 450) &
                                                     (self.master_dataset.loc[:, 'temperature'] != 400) &
                                                     (self.master_dataset.loc[:, 'temperature'] != 350) &
                                                     (self.master_dataset.loc[:, 'temperature'] != 150)].index,
                }

                temp_slice = temp_dict.get(temp_filter)
            else:
                temp_slice = self.master_dataset[self.master_dataset.loc[:, 'temperature'] == temp_filter].index
            return temp_slice

        def filter_ammonia(ammo_filter):
            filter_dict_ammonia = {
                1: self.master_dataset[(self.master_dataset.loc[:, 'ammonia_concentration'] > 0.5) &
                                       (self.master_dataset.loc[:, 'ammonia_concentration'] < 1.5)].index,
                5: self.master_dataset[(self.master_dataset.loc[:, 'ammonia_concentration'] > 4.8) &
                                       (self.master_dataset.loc[:, 'ammonia_concentration'] < 5.2)].index
            }

            ammo_slice = filter_dict_ammonia.get(ammo_filter, self.master_dataset.index)
            return ammo_slice

        def filter_ruthenium_weight_loading(ru_filter):
            filter_dict_ruthenium = {
                1: self.master_dataset[(self.master_dataset.loc[:, 'Ru Loading'] == 0.01)].index,
                2: self.master_dataset[(self.master_dataset.loc[:, 'Ru Loading'] == 0.02)].index,
                3: self.master_dataset[(self.master_dataset.loc[:, 'Ru Loading'] == 0.03)].index,
            }

            ru_slice = filter_dict_ruthenium.get(ru_filter, self.master_dataset.index)
            return ru_slice

        def filter_pressure(p_filter):
            filter_dict_ruthenium = {

            }

            p_slice = filter_dict_ruthenium.get(p_filter, self.master_dataset.index)
            return p_slice

        def filter_space_velocity(sv_filter):
            filter_dict_sv = {
                2000: self.master_dataset[(self.master_dataset.loc[:, 'space_velocity'] > 1400) &
                                       (self.master_dataset.loc[:, 'space_velocity'] < 2600)].index
            }
            sv_slice = filter_dict_sv.get(sv_filter, self.master_dataset.index)
            return sv_slice

        def drop_element(ele, filt):
            slice = self.master_dataset[self.master_dataset['{} Loading'.format(ele)] == 0].index
            return join_all_indecies([filt, slice])

        def drop_ID(ID, filt):
            indx = ['{id}_{temp}'.format(id=x.split('_')[0], temp=x.split('_')[1]) != ID for x in self.master_dataset.index]
            slice = self.master_dataset[indx].index
            return join_all_indecies([filt, slice])

        filt = join_all_indecies([
            filter_elements(self.element_filter),
            filter_temperature(self.temperature_filter),
            filter_ammonia(self.ammonia_filter),
            filter_space_velocity(self.sv_filter),
            filter_ruthenium_weight_loading(self.ru_filter),
            filter_pressure(self.pressure_filter)
        ])

        # filt = drop_element('Y', filt)
        # filt = drop_element('Rh', filt)
        # filt = drop_element('Hf', filt)

        # ***** FILTER SS8 AND SS9 OUT *****
        # for ids in [65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83, 38, 84, 85, 86, 87, 89, 90, 91, 93]:
        #     for tempers in [250, 300, 350, 400, 450]:
        #         filt = drop_ID('{id}_{t}'.format(id=ids, t=tempers), filt)

        # Apply filter, master to slave, shuffle slave
        self.slave_dataset = self.master_dataset.loc[filt].copy()
        self.slave_dataset = self.slave_dataset[self.slave_dataset.index.str.contains('Predict') == False]
        self.slave_dataset = shuffle(self.slave_dataset)
        pd.DataFrame(self.slave_dataset).to_csv('..\\SlaveTest.csv')

        # Set up training data and apply grouping
        self.set_training_data()
        self.group_for_training()
        # self.trim_slave_dataset()

    def drop_features(self):
        pass

    def set_training_data(self):
        ''' Use the slave dataframe to set other dataframe properties '''

        if self.average_data:
            self.features_df = self.slave_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged', 'group'], axis=1
            )
        else:
            self.features_df = self.slave_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary'], axis=1
            )

        if self.regression:
            self.labels_df = self.slave_dataset['Measured Conversion'].copy()
        else:
            self.labels_df = self.slave_dataset['Measured Conversion'].copy()
            self.labels_df[self.labels_df >= 0.8] = 1
            self.labels_df[self.labels_df < 0.8] = 0

    def group_for_training(self):
        """ Comment """

        group_dict = {
            'full-blind': self.slave_dataset['group'].values,
            'blind': [x.split('_')[0] for x in self.slave_dataset.index.values],
            'semiblind': ['{}_{}'.format(x.split('_')[0], x.split('_')[1]) for x in self.slave_dataset.index.values]
        }

        self.groups = group_dict.get(self.group_style, None)

    def trim_slave_dataset(self):
        ''' Feature Selection '''
        # trim = VarianceThreshold(threshold=0.9)
        # test = trim.fit_transform(self.features_df.values, self.labels_df.values)
        # self.features_df = pd.DataFrame(trim.inverse_transform(test), columns=self.features_df.columns, index=self.features_df.index)
        # self.features_df = self.features_df.loc[:, (self.features_df != 0).any(axis=0)]
        self.features_df = self.features_df.loc[
                           :, ['temperature',
                               'Number d-shell Valence Electrons_wt-mean', 'Number d-shell Valence Electrons_wt-mad',
                               ]
                           ].copy()

    def hyperparameter_tuning(self):
        """ Comment """
        rfr_tuning_params = {
            'n_estimators': [10, 25, 50, 100],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_leaf_nodes': [None, 5, 20, 50],
            'min_impurity_decrease': [0, 0.1, 0.4]
        }

        gbr_tuning_params = {
            'loss': ['ls', 'lad', 'quantile', 'huber'],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.5, 1],
            'n_estimators': [25, 100, 500],
            'max_depth': [None, 3, 5, 10],
            'criterion': ['friedman_mse', 'mae'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': ['auto', 'sqrt'],
            'max_leaf_nodes': [None, 5, 20, 50],
            'min_impurity_decrease': [0, 0.1, 0.4]
        }

        etr_tuning_params = {
            'n_estimators': [10, 25, 50, 100, 200, 400],
            'criterion': ['mae'],
            'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.1, 0.05, 0.01],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_leaf_nodes': [None, 5, 20, 50],
            'min_impurity_decrease': [0, 0.1, 0.4]
        }

        # gs = GridSearchCV(self.machina, self.machina_tuning_parameters, cv=10, return_train_score=True)
        gs = RandomizedSearchCV(self.machina, etr_tuning_params, cv=GroupKFold(3),
                                return_train_score=True, n_iter=1000)
        gs.fit(self.features_df.values, self.labels_df.values, groups=self.groups)
        pd.DataFrame(gs.cv_results_).to_csv('{}\\p-tune-gbr_{}.csv'.format(self.svfl, self.svnm))

    def set_learner(self, learner, params='default'):
        """ Comment """
        learn_selector = {
            # Regression Models
            'rfr': RandomForestRegressor,
            'adaboost': AdaBoostRegressor,
            'tree': tree.DecisionTreeRegressor,
            'SGD': SGDRegressor,
            'neuralnet': MLPRegressor,
            'svr': SVR,
            'knnr': KNeighborsRegressor,
            'krr': KernelRidge,
            'etr': ExtraTreesRegressor,
            'gbr': GradientBoostingRegressor,
            'ridge': Ridge,
            'lasso': Lasso,

            # Classification Models
            'rfc': RandomForestClassifier,

            # Others
        }

        if self.regression:
            param_selector = {
                'rfr': {'n_estimators':25, 'max_depth':10, 'max_leaf_nodes':50, 'min_samples_leaf':1,
                            'min_samples_split':2, 'max_features':'auto', 'bootstrap':True, 'n_jobs':4,
                            'criterion':'mae'},
                'etr': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_impurity_decrease': 0,
                        'max_leaf_nodes': 50, 'max_features': 'auto', 'max_depth': 10, 'criterion': 'mae'},
                'etr-old': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_impurity_decrease': 0,
                        'max_leaf_nodes': None, 'max_features': 'sqrt', 'max_depth': 10, 'criterion': 'mae'},
                'gbr': {'subsample': 0.5, 'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 3,
                        'min_impurity_decrease': 0, 'max_leaf_nodes': 5, 'max_features': 'sqrt', 'max_depth': 5,
                        'loss': 'ls', 'learning_rate': 0.05, 'criterion': 'mae'},
                'adaboost': {'base_estimator':RandomForestRegressor(), 'n_estimators':1000},
                'nnet': {'hidden_layer_sizes':1, 'solver':'lbfgs'},
                'empty': {},
                'SGD': {'alpha': 0.01, 'tol': 1e-4, 'max_iter': 1000}
            }
        else:
            param_selector = {
                'default': {'n_estimators':100, 'n_jobs':4}
            }

        self.machina = learn_selector.get(learner, lambda: 'Error')()
        self.machina.set_params(**param_selector.get(params))

    def unsupervised_data_segmentation(self, n_clusters=3):
        """ Pre-Cluster Data to segregate chemically similar catalysts """
        self.n_clusters = n_clusters
        # self.cluster = KMeans(n_clusters=self.n_clusters)
        self.cluster = AgglomerativeClustering(n_clusters=n_clusters)
        feats = self.features_df.drop(columns=['temperature',
                                               'space_velocity',
                                               'reactor_number',
                                               'ammonia_concentration']).values
        self.slave_dataset['kmeans'] = self.cluster.fit_predict(feats)
        print(self.cluster.labels_)
        self.slave_dataset.to_csv('{}//{}-unsupervised-slave-dataset.csv'.format(self.svfl, self.version))

    def train_data(self):
        """ Comment """

        if self.cluster is None:
            self.machina = self.machina.fit(self.features_df.values, self.labels_df.values)
        else:
            for n in range(self.n_clusters):
                print(n)
                idx = self.slave_dataset[self.slave_dataset['kmeans'] == n].index
                self.clustered_machina_dictionary[n] = self.machina.fit(
                    self.features_df.loc[idx].values,
                    self.labels_df.loc[idx].values
                )

    def create_test_dataset(self, catids):
        """ Description """

        # Create Temporary indexer to slice slave dataset
        ind = [int(idtag.split('_')[0]) for idtag in self.slave_dataset.index]
        self.slave_dataset['temporary_ind'] = ind

        # Slice the dataset, copying all values of catids
        self.test_dataset = self.slave_dataset[self.slave_dataset['temporary_ind'].isin(catids)].copy()

        # Drop the temporary indexer
        self.slave_dataset.drop(columns=['temporary_ind'], inplace=True)
        self.test_dataset.drop(columns=['temporary_ind'], inplace=True)

        # Remove test dataset from slave dataset to prepare for training
        self.slave_dataset.drop(labels=self.test_dataset.index, inplace=True)

        self.set_training_data()

    def predict_from_masterfile(self, catids, svnm='data', temp_slice=True):
        """ Descr """
        self.create_test_dataset(catids)
        self.train_data()

        """ Comment - Work in Progress """
        if self.average_data:
            data = self.test_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged', 'group'],
                axis=1).values
        else:
            data = self.test_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary'],
                axis=1).values

        predvals = self.machina.predict(data)

        original_test_df = self.master_dataset.loc[self.test_dataset.index].copy()
        measvals = original_test_df.loc[:, 'Measured Conversion'].values

        comparison_df = pd.DataFrame([predvals, measvals],
                           index=['Predicted Conversion','Measured Conversion'],
                           columns=original_test_df.index).T

        comparison_df['ID'] = [x.split('_')[0] for x in comparison_df.index]
        comparison_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in self.test_dataset['Element Dictionary']
        ]
        comparison_df['temperature'] = original_test_df['temperature']
        comparison_df.drop(comparison_df[comparison_df.loc[:, 'temperature'] == 450].index, inplace=True)
        comparison_df.drop(comparison_df[comparison_df.loc[:, 'temperature'] == 400].index, inplace=True)

        feat_df = self.extract_important_features()
        feat_df.to_csv('{}\\{}-features.csv'.format(self.svfl, svnm))

        g = Graphic(learner=self, df=comparison_df)
        g.plot_err(svnm='{}-predict_{}'.format(self.version, svnm))
        g.plot_err(svnm='{}-predict_{}_nometa'.format(self.version, svnm), metadata=False)
        g.plot_important_features(svnm=svnm)
        g.bokeh_predictions(svnm='{}-predict_{}'.format(self.version, svnm))

    def predict_dataset(self):
        data = self.master_dataset[self.master_dataset.index.str.contains('Predict') == True]
        data = data.drop(
            labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged', 'group'], axis=1
        )

        predvals = self.machina.predict(data.values)
        data['Predictions'] = predvals
        data.to_csv(r'{}/{}-BinaryPredictions.csv'.format(self.svfl, self.version))
        return data

    def predict_crossvalidate(self, kfold=10):
        """ Comment """
        if self.cluster is None:
            self.predictions = cross_val_predict(self.machina, self.features_df.values, self.labels_df.values,
                                             groups=self.groups, cv=GroupKFold(kfold))

            self.slave_dataset['predictions'] = self.predictions

        else:
            group_df = pd.DataFrame(self.groups, index=self.slave_dataset.index)
            preddf = pd.DataFrame()

            for n in range(self.n_clusters):
                idx = self.slave_dataset[self.slave_dataset['kmeans'] == n].index

                if group_df.loc[idx, 0].unique().size < kfold:
                    k = group_df.loc[idx, 0].unique().size
                else:
                    k = kfold

                pred = cross_val_predict(estimator=self.machina,
                                         X=self.features_df.loc[idx].values,
                                         y=self.labels_df.loc[idx].values,
                                         groups=group_df.loc[idx].values,
                                         cv=GroupKFold(k)
                                         )

                preddf = pd.concat([preddf, pd.DataFrame(pred, index=idx)])

            self.slave_dataset['predictions'] = preddf
            self.predictions = self.slave_dataset['predictions'].values

        self.slave_dataset.to_csv('{}//{}-slave.csv'.format(self.svfl, self.svnm))

    def preplot_processing(self):
        """ Prepare all data for plotting """

        # Ensure Predictions Exist
        if self.predictions is None:
            self.predict_crossvalidate()

        # Set up the plot dataframe for easy plotting
        self.plot_df = self.slave_dataset.copy()
        self.plot_df['Predicted Conversion'] = self.predictions

        self.plot_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in self.plot_df['Element Dictionary']
        ]

        for index, edict in self.plot_df['Element Dictionary'].iteritems():
            self.plot_df.loc[index, 'Name'] = ''.join('{}({})'.format(key, str(int(val))) for key, val in edict)

            i = 1
            for key, val in edict:
                self.plot_df.loc[index, 'Ele{}'.format(i)] = key
                self.plot_df.loc[index, 'Load{}'.format(i)] = val
                i += 1

        self.save_predictions()

    def save_predictions(self):
        """ Comment """
        if not self.plot_df.empty:
            df = pd.DataFrame(np.array([self.plot_df.index, self.predictions, self.labels_df.values, self.groups, self.plot_df['Name']]).T,
                              columns=['ID', 'Predicted Conversion', 'Measured Conversion', 'Groups', 'Name'])
            df.to_csv('{}\predictions-{}.csv'.format(self.svfl, self.svnm))
        else:
            print('No predictions to save...')

    def extract_important_features(self, sv=False, prnt=False):
        """ Save all feature importance, print top 10 """

        try:
            df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns,
                          columns=['Feature Importance'])

            self.feature_importance_df = df
        except AttributeError:
            return

        if prnt:
            print(df.sort_values(by='Feature Importance', ascending=False).head(10))

        if sv:
            df.to_csv('{}//features//feature_importance-{}.csv'.format(self.svfl, self.svnm))

            new_df = pd.DataFrame()

            for nm in df.index:
                df.loc[nm, 'Feature'] = nm.split('_')[0]

            for feat in df.Feature.unique():
                new_df.loc[feat, 'Feature Importance'] = df[df['Feature'] == feat]['Feature Importance'].sum()

            new_df.sort_values('Feature Importance', ascending=False, inplace=True)
            new_df.to_csv('{}//features//feature_importance-{}-summed.csv'.format(self.svfl, self.svnm))
        else:
            return df



    def evaluate_regression_learner(self):
        """ Comment """
        r2 = r2_score(self.labels_df.values, self.predictions)
        mean_abs_err = mean_absolute_error(self.labels_df.values, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.labels_df.values, self.predictions))

        print('\n----- Model {} -----'.format(self.svnm))
        print('R2: {:0.3f}'.format(r2))
        print('Mean Average Error: {:0.3f}'.format(mean_abs_err))
        print('Mean Squared Error: {:0.3f}'.format(rmse))
        print('Time to Complete: {:0.1f} s'.format(time.time() - self.start_time))
        print('\n')

        pd.DataFrame([r2, mean_abs_err, rmse, time.time() - self.start_time],
                     index=['R2','Mean Abs Error','Root Mean Squared Error','Time']
                     ).to_csv('{}\\eval\\{}-eval.csv'.format(self.svfl, self.svnm))

    def evaluate_classification_learner(self):
        print(self.predictions)
        fpr, tpr, thershold = roc_curve(self.labels_df.values, self.predictions)
        print(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.show()



        #
        # # Full descriptive name X(#)Y(#)Z(#)
        # self.plot_df['Name'] = [
        #     ''.join('{}({})'.format(key, str(int(val)))
        #             for key, val in x) for x in self.plot_df['Element Dictionary']
        # ]
        #
        # # Second Element Name
        # self.plot_df['Ele2'] = [
        #     ''.join('{}'.format(key) if (key != 'Ru') & (key != 'K') else ''
        #             for key, val in x) for x in self.plot_df['Element Dictionary']
        # ]
        #
        # # Second Element Weight Loading
        # self.plot_df['Load2'] = [
        #     ''.join('{}'.format(val) if (key != 'Ru') & (key != 'K') else ''
        #             for key, val in x) for x in self.plot_df['Element Dictionary']
        # ]
        #
        # # CatalystObject ID
        # self.plot_df['ID'] = [int(nm.split('_')[0]) for nm in self.plot_df.index.values]
        #
        # # Remove Dictionary to avoid problems down the line
        # self.plot_df.drop(columns='Element Dictionary', inplace=True)
        #
        # # Create hues for heatmaps
        # def create_feature_hues(self, feature):
        #     try:
        #         unique_feature = np.unique(self.slave_dataset.loc[:, feature].values)
        #     except KeyError:
        #         print('KeyError: {} not found'.format(feature))
        #         return
        #
        #     n_feature = len(unique_feature)
        #     max_feature = np.max(unique_feature)
        #     min_feature = np.min(unique_feature)
        #
        #     if max_feature == min_feature:
        #         self.plot_df['{}_hues'.format(feature)] = "#3498db"  # Blue!
        #     else:
        #         palette = sns.color_palette('plasma', n_colors=n_feature+1)
        #         self.plot_df['{}_hues'.format(feature)] = [
        #             palette[i] for i in [int(n_feature * (float(x) - min_feature) / (max_feature - min_feature))
        #                                       for x in self.slave_dataset.loc[:, feature].values]
        #         ]
        #
        # self.plot_df['temperature_hues'] = 0
        #
        # # Grab top 10 features, add hues to plotdf
        # try:
        #     feature_rank = self.extract_important_features()
        #     for feat in feature_rank.sort_values(by='Feature Importance', ascending=False).head(10).index.values:
        #         create_feature_hues(self, feat)
        # except AttributeError:
        #     print('Learner does not support feature extraction.')
        #
        # # Process Second Element Colors
        # uniq_eles = np.unique(self.plot_df['Ele2'])
        # n_uniq = len(uniq_eles)
        # palette = sns.color_palette('tab20', n_colors=n_uniq + 1)
        # self.plot_df['Ele2_hues'] = [
        #     palette[np.where(uniq_eles == i)[0][0]] for i in self.plot_df['Ele2']
        # ]
        #
        # return self.plot_df

    def plot_basic(self):
        """ Creates a basic plot with a temperature heatmap (or constant color if single temp slice) """

        df = pd.DataFrame([self.predictions,
                           self.labels_df.values,
                           self.plot_df['{}_hues'.format('temperature')].values,
                           self.plot_df['{}'.format('temperature')].values],
                          index=['pred', 'meas', 'clr', 'temperature']).T

        uniq_temps = np.unique(df['temperature'])

        for feat in uniq_temps:
            plt.scatter(x=df.loc[df['temperature'] == feat, 'pred'],
                        y=df.loc[df['temperature'] == feat, 'meas'],
                        c=df.loc[df['temperature'] == feat, 'clr'],
                        label='{}{}C'.format(int(feat), u'\N{DEGREE SIGN}'),
                        edgecolors='k')

        plt.xlabel('Predicted Conversion')
        plt.ylabel('Measured Conversion')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend(title='Temperature')
        plt.tight_layout()

        if len(uniq_temps) == 1:
            plt.savefig('{}//figures//{}-basic-{}C.png'.format(self.svfl, self.svnm, uniq_temps[0]),
                        dpi=400)
        else:
            plt.savefig('{}//figures//{}-basic.png'.format(self.svfl, self.svnm),
                    dpi=400)
        plt.close()

    def plot_error(self, metadata=True):
        df = pd.DataFrame([self.predictions,
                           self.labels_df.values,
                           self.plot_df['{}_hues'.format('temperature')].values,
                           self.plot_df['{}'.format('temperature')].values],
                          index=['pred', 'meas', 'clr', 'feat']).T

        rats = np.abs(np.subtract(self.predictions, self.labels_df.values, out=np.zeros_like(self.predictions),
                         where=self.labels_df.values != 0))

        rat_count = rats.size
        wi5 = (rats < 0.05).sum()
        wi10 = (rats < 0.10).sum()
        wi20 = (rats < 0.20).sum()

        fig, ax = plt.subplots()

        uniq_features = np.unique(df['feat'])

        # Katie's Colors
        # color_selector = {
        #     250: 'purple',
        #     300: 'darkgreen',
        #     350: 'xkcd:coral',
        #     400: 'darkblue',
        #     450: 'xkcd:salmon'
        # }

        cmap = get_cmap('plasma')
        norm = Normalize(vmin=250, vmax=450)

        color_selector = {
            250: cmap(norm(250)),
            300: cmap(norm(300)),
            350: cmap(norm(350)),
            400: cmap(norm(400)),
            450: cmap(norm(450))
        }

        if len(uniq_features) == 1:


            ax.scatter(x=df.loc[df['feat'] == uniq_features[0], 'pred'],
                       y=df.loc[df['feat'] == uniq_features[0], 'meas'],
                       c=color_selector.get(uniq_features[0]),
                       label='{}{}C'.format(int(uniq_features[0]), u'\N{DEGREE SIGN}'),
                       edgecolors='k')
        else:
            for feat in uniq_features:
                ax.scatter(x=df.loc[df['feat'] == feat, 'pred'],
                           y=df.loc[df['feat'] == feat, 'meas'],
                           c=color_selector.get(feat),  #df.loc[df['feat'] == feat, 'clr'],
                           label='{}{}C'.format(int(feat), u'\N{DEGREE SIGN}'),
                           edgecolors='k')

        x = np.array([0, 0.5, 1])
        y = np.array([0, 0.5, 1])

        ax.plot(x, y, lw=2, c='k')
        ax.fill_between(x, y + 0.1, y - 0.1, alpha=0.1, color='b')
        ax.fill_between(x, y + 0.2, y + 0.1, alpha=0.1, color='y')
        ax.fill_between(x, y - 0.2, y - 0.1, alpha=0.1, color='y')

        if metadata:
            plt.figtext(0.99, 0.01, 'Within 5%: {five:0.2f} \nWithin 10%: {ten:0.2f} \nWithin 20%: {twenty:0.2f}'.format(
                five=wi5 / rat_count, ten=wi10 / rat_count, twenty=wi20 / rat_count),
                        horizontalalignment='right', fontsize=6)

            mean_abs_err = mean_absolute_error(self.labels_df.values, self.predictions)
            rmse = np.sqrt(mean_squared_error(self.labels_df.values, self.predictions))

            plt.figtext(0, 0.01, 'MeanAbsErr: {:0.2f} \nRMSE: {:0.2f}'.format(mean_abs_err, rmse),
                        horizontalalignment='left', fontsize=6)

            plt.figtext(0.5, 0.01, 'E{} A{} S{}'.format(self.element_filter, self.ammonia_filter, self.sv_filter),
                        horizontalalignment='center', fontsize=6)

        plt.xlabel('Predicted Conversion')
        plt.ylabel('Measured Conversion')
        plt.legend(title='Temperature')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.title('{}-{}'.format(self.svnm, 'err'))
        plt.legend(title='Temperature')
        plt.tight_layout()
        plt.savefig('{}//figures//{}-{}.png'.format(self.svfl, self.svnm, 'err'),
                    dpi=400)
        plt.close()

    def plot_features(self, x_feature, c_feature):
        uniqvals = np.unique(self.plot_df[c_feature].values)
        for cval in uniqvals:
            slice = self.plot_df[c_feature] == cval
            plt.scatter(x=self.plot_df.loc[slice, x_feature], y=self.plot_df.loc[slice, 'Measured Conversion'],
                        c=self.plot_df.loc[slice, '{}_hues'.format(c_feature)], label=cval, s=30, edgecolors='k')
        plt.xlabel(x_feature)
        plt.ylabel('Measured Conversion')
        plt.ylim(0, 1)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig('{}//figures//{}-x({})-c({}).png'.format(self.svfl, self.svnm, x_feature, c_feature), dpi=400)
        plt.close()

    def plot_features_colorbar(self, x_feature, c_feature):
        plt.scatter(x=self.plot_df.loc[:, x_feature], y=self.plot_df.loc[:, 'Measured Conversion'],
                    c=self.plot_df.loc[:, '{}'.format(c_feature)], cmap='viridis', s=30, edgecolors='k')
        plt.xlabel(x_feature)
        plt.ylabel('Measured Conversion')
        plt.ylim(0, 1)
        cbar = plt.colorbar()
        cbar.set_label(c_feature)
        plt.tight_layout()
        plt.savefig('{}//figures//{}-x({})-c({}).png'.format(self.svfl, self.svnm, x_feature, c_feature), dpi=400)
        plt.close()

    def plot_important_features(self):
        """ Comment """
        featdf = self.extract_important_features()
        top5feats = featdf.nlargest(5, 'Feature Importance').index.values.tolist()
        feats = self.slave_dataset.loc[:, top5feats+['Measured Conversion']]
        feats['hue'] = np.ceil(feats['Measured Conversion'].values * 5)
        sns.pairplot(feats, hue='temperature', diag_kind='kde')
        plt.tight_layout()
        plt.savefig('{}//figures//{}-featrels.png'.format(self.svfl, self.svnm))
        plt.close()

    def bokeh_predictions(self):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@temperature')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        if self.plot_df['temperature_hues'].all() != 0:
            self.plot_df['bokeh_color'] = self.plot_df['temperature_hues'].apply(rgb2hex)
        else:
            self.plot_df['bokeh_color'] = 'blue'

        source = ColumnDataSource(self.plot_df)

        p.circle("Predicted Conversion", "Measured Conversion", size=12, source=source,
                 color='bokeh_color', line_color="black", fill_alpha=0.8)

        output_file("{}\\htmls\\{}.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def bokeh_by_elements(self):
        """ HTML with overview with colorscheme that is per-element """
        if self.predictions is None:
            self.predict_crossvalidate()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@temperature')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        self.plot_df['bokeh_color'] = self.plot_df['Ele2_hues'].apply(rgb2hex)

        source = ColumnDataSource(self.plot_df)

        p.circle("Predicted Conversion", "Measured Conversion", size=12, source=source,
                 color='bokeh_color', line_color="black", fill_alpha=0.8)

        output_file("{}\\htmls\\{}_byeles.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def bokeh_averaged(self, whiskers=False):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        df = pd.DataFrame(np.array([
            [int(nm.split('_')[0]) for nm in self.slave_dataset.index.values],
            self.predictions,
            self.labels_df.values,
            self.slave_dataset.loc[:, 'temperature'].values]).T,
                          columns=['ID', 'Predicted', 'Measured', 'Temperature'])

        cat_eles = self.slave_dataset.loc[:, 'Element Dictionary']
        vals = [''.join('{}({})'.format(key, str(int(val))) for key, val in x) for x in cat_eles]
        df['Name'] = vals

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@Temperature')
        ])
        tools.append(hover)

        unique_temps = len(df['Temperature'].unique())
        max_temp = df['Temperature'].max()
        min_temp = df['Temperature'].min()

        if max_temp == min_temp:
            df['color'] = pals.plasma(5)[4]
        else:
            pal = pals.plasma(unique_temps + 1)
            df['color'] = [pal[i]
                           for i in [int(unique_temps * (float(x) - min_temp) / (max_temp - min_temp))
                                     for x in df['Temperature']]]

        unique_names = np.unique(df.loc[:, 'Name'].values)

        final_df = pd.DataFrame()

        for nm in unique_names:
            nmdf = df.loc[df.loc[:, 'Name'] == nm]
            unique_temp = np.unique(nmdf.loc[:, 'Temperature'].values)

            for temperature in unique_temp:
                tdf = nmdf.loc[nmdf.loc[:, 'Temperature'] == temperature]
                add_df = tdf.iloc[0, :].copy()
                add_df['Measured'] = tdf['Measured'].mean()
                add_df['Measured Standard Error'] = tdf['Measured'].sem()
                add_df['Upper'] = tdf['Measured'].mean() + tdf['Measured'].sem()
                add_df['Lower'] = tdf['Measured'].mean() - tdf['Measured'].sem()
                add_df['n Samples'] = tdf['Measured'].count()

                final_df = pd.concat([final_df, add_df], axis=1)

        df = final_df.transpose()

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(df)

        p.circle("Predicted", "Measured", size=8, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        if whiskers:
            p.add_layout(
                Whisker(source=source, base="Predicted", upper="Upper", lower="Lower", level="overlay")
            )

        output_file("{}\\{}_avg.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def bokeh_important_features(self, svtag='IonEn',
                                 xaxis='Measured Conversion', xlabel='Measured Conversion', xrng=None,
                                 yaxis='Predicted Conversion', ylabel='Predicted Conversion', yrng=None,
                                 caxis='temperature'
                                 ):
        """ Comment """

        # uniqvals = np.unique(self.plot_df[caxis].values)
        # for cval in uniqvals:
        #     slice = self.plot_df[caxis] == cval
        #     plt.scatter(x=self.plot_df.loc[slice, xaxis], y=self.plot_df.loc[slice, yaxis],
        #                 c=self.plot_df.loc[slice, '{}_hues'.format(caxis)], label=cval, s=30, edgecolors='k')

        # unique_temps = len(featdf['temperature'].unique())
        # max_temp = featdf['temperature'].max()
        # min_temp = featdf['temperature'].min()
        #
        # if max_temp == min_temp:
        #     featdf['color'] = pals.plasma(5)[4]
        # else:
        #     pal = pals.plasma(unique_temps + 1)
        #     featdf['color'] = [pal[i]
        #                    for i in [int(unique_temps * (float(x) - min_temp) / (max_temp - min_temp))
        #                              for x in featdf['temperature']]]
        #
        # if temp_slice is not None:
        #     featdf = featdf[featdf['temperature'] == temp_slice]
        #
        if xrng is None:
            xrng = DataRange1d()
        if yrng is None:
            yrng = DataRange1d()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('IonEn', '@IonizationEnergies_2_1')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = xrng
        p.y_range = yrng
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        p.grid.grid_line_color = "white"

        try:
            self.plot_df['bokeh_color'] = self.plot_df['{}_hues'.format(caxis)].apply(rgb2hex)
        except KeyError:
            self.plot_df['bokeh_color'] = 'blue'

        source = ColumnDataSource(self.plot_df)
        p.circle(xaxis, yaxis, size=12, source=source,
                 color='bokeh_color', line_color="black", fill_alpha=0.8)

        output_file("{}\\{}{}.html".format(self.svfl, self.svnm, '-{}'.format(svtag) if svtag is not '' else ''), title="stats.py")
        save(p)

    def bokeh_test(self):
        output_file("callback.html")

        self.plot_df['x'] = self.plot_df['Predicted Conversion'].values
        source = ColumnDataSource(self.plot_df)
        p = figure(plot_width=400, plot_height=400)
        p.circle('x', 'Measured Conversion', source=source)

        def callback(src=source, window=None):
            data = src.data
            data['x'] = data[cb.obj.value].values
            source.change.emit()

        x = Select(options=self.plot_df.columns.tolist(), title='X Axis', callback=CustomJS.from_py_func(callback))
        y = Select(options=self.plot_df.columns.tolist(), title='Y Axis')
        c = Select(options=self.plot_df.columns.tolist(), title='Color')
        # s = Select(options=self.plot_df.values.tolist())
        ly = layout(p, x, y, c)

        show(ly)

    def visualize_tree(self, n=1):
        """ Comment """
        if n == 1:
            gv = tree.export_graphviz(self.machina.estimators_[0],
                                      filled=True,
                                      out_file='{}//Trees//{}.dot'.format(self.svfl, self.svnm),
                                      feature_names=self.features_df.columns,
                                      rounded=True)

            os.system('dot -Tpng {fl}//Trees//{nm}.dot -o {fl}//Trees//{nm}_singtree.png'.format(fl=self.svfl,
                                                                                                 nm=self.svnm))

        else:
            for index, forest in enumerate(self.machina.estimators_):
                gv = tree.export_graphviz(forest,
                                          filled=True,
                                          out_file='{}//Trees//{}.dot'.format(self.svfl, self.svnm),
                                          feature_names=self.features_df.columns,
                                          rounded=True)

                os.system('dot -Tpng {fl}//Trees//{nm}.dot -o {fl}//Trees//{nm}-{ind}.png'.format(fl=self.svfl,
                                                                                                  nm=self.svnm,
                                                                                                  ind=index))

    def parse_element_dictionary(self):
        df = self.master_dataset[['Element Dictionary']].copy()
        df['index'] = [idx.split('_')[0] for idx in df.index]
        things = [list(x) for x in df['Element Dictionary'].values]
        for idx, elements in df[['Element Dictionary']]:
            print(idx, elements)
            df[idx, 'ele1'] = elements[0][0]
            df[idx, 'ele2'] = elements[1][0]
            df[idx, 'ele3'] = elements[2][0]
            df[idx, 'wt1'] = elements[0][1]
            df[idx, 'wt2'] = elements[1][1]
            df[idx, 'wt3'] = elements[2][1]

        print(df)

        exit()
