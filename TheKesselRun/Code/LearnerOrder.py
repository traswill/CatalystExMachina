# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

from TheKesselRun.Code.Plotter import Graphic
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.feature_selection import SelectKBest, RFECV, RFE
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel, chi2_kernel


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, GroupKFold, LeaveOneGroupOut, \
    LeaveOneOut, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.utils import shuffle


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

    def build_master_container(self, drop_empty_columns=True):
        # Set up catalyst loading dictionary with loadings
        loading_df = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation'], index_col='Abbreviation').transpose()
        loading_df.columns = ['{} Loading'.format(ele) for ele in loading_df.columns]

        for catid, catobj in self.catalyst_dictionary.items():
            # Reset loading dictionary
            load_df = loading_df.copy()

            # Add elements and loading to loading dict
            for ele, wt in catobj.elements_wt.items():
                load_df.loc[catid, '{} Loading'.format(ele)] = wt / 100

            for ele, mol in catobj.elements_mol.items():
                load_df.loc[catid, '{} mol%'.format(ele)] = mol / 100

            # Create group
            groupdf = pd.DataFrame(catobj.group, index=[catid], columns=['group'])

            # Create DF from features
            featdf = pd.DataFrame.from_dict(catobj.feature_dict, orient='index').transpose()
            featdf.index = [catid]

            # Create element dictionary
            eldictdf = pd.DataFrame(catobj.elements_wt.items(), index=[catid], columns=['Element Dictionary'])

            df = pd.concat([load_df, featdf, eldictdf, groupdf], axis=1)

            # Iterate through observations and add catalysts
            for key, obs in catobj.observation_dict.items():
                inputdf = pd.DataFrame.from_dict(obs.to_dict(), orient='index').transpose()
                inputdf.index = [catid]

                catdf = pd.concat([df, inputdf], axis=1)
                self.master_container = pd.concat([self.master_container, catdf], axis=0, sort=True)

        if drop_empty_columns:
            self.master_container.dropna(how='all', axis=1, inplace=True)
        self.master_container.fillna(value=0, inplace=True)

        # This code overwrites the groups provided to the ML model to force all similar-element catalyst into the same group
        df = pd.DataFrame(sorted([ele[0] for ele in list(x)]) for x in self.master_container['Element Dictionary'].values)
        df['edict'] = self.master_container['Element Dictionary'].values
        df.fillna('', inplace=True)
        df['group'] = df.groupby([0,1,2]).ngroup()
        self.master_container['group'] = df['group'].values

        # Transfer catalyst ID to column so each index is unique
        self.master_container['ID'] = self.master_container.index
        self.master_container.reset_index(inplace=True, drop=True)

class SupervisedLearner():
    """SupervisedLearner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self, version='v00', note=None):
        """ Initialize Everything """

        '''Initialize Main Dataframes'''
        self.static_dataset = pd.DataFrame() # Dataset that is never changed and used to reset
        self.dynamic_dataset = pd.DataFrame() # Dataset that is always used as the working dataset
        self.result_dataset = pd.DataFrame() # Dataset for use after testing model

        '''Initialize Column Identifiers'''
        self.target_columns = list() # list of columns in master_dataset with target values to be predicted
        self.group_columns = list() # list of column in master_dataset to use for grouping catalysts
        self.hold_columns = list() # list of columns to remove from the feature set during training
        self.drop_columns = list() # features to drop from training dataset permanently

        '''Initialize Sub Dataframes'''
        self.hold_df = pd.DataFrame()
        self.features_df = pd.DataFrame()
        self.labels_df = pd.DataFrame()

        '''Initialize Variables'''
        self.features = np.empty(1)
        self.labels = np.empty(1)
        self.groups = np.empty(1)
        self.predictions = list()
        self.tau = 0.
        self.uncertainty = list()

        '''Initialize ML algorithm'''
        self.machina = None

        '''Initialize all options for the algorithm.  These are used in naming files.'''
        self.num_element_filter = 0
        self.temperature_filter = None
        self.ammonia_filter = None
        self.ru_filter = None
        self.pressure_filter = None
        self.sv_filter = None
        self.promoter_filter = None
        self.version = version

        '''Initialize and create path, folder, and filename'''
        self.svfl = '..//Results//{version}'.format(version=version)
        self.svnm = '{nm}-{nele}-{temp}'.format(
            nm=version,
            nele=self.num_element_filter,
            temp='{}C'.format(self.temperature_filter) if self.temperature_filter is not None else 'All'
        )

        if not os.path.exists(self.svfl):
            os.makedirs(self.svfl)
            os.makedirs('{}\\{}'.format(self.svfl, 'trees'))
            os.makedirs('{}\\{}'.format(self.svfl, 'figures'))
            os.makedirs('{}\\{}'.format(self.svfl, 'htmls'))
            os.makedirs('{}\\{}'.format(self.svfl, 'features'))
            os.makedirs('{}\\{}'.format(self.svfl, 'eval'))

        ''' Add Note text file if applicable '''
        if note:
            with open('{}\\readme.txt'.format(self.svfl), 'w') as txtfl:
                print(note, file=txtfl)

        '''Initialize Time for run-length statistics'''
        self.start_time = time.time()

    def set_name_paths(self):
        """ These paths are used by all methods to save files to the proper location.  This method is used to reset
         the save directories in the event of changes to the version or other variables.
         """

        self.svfl = '..//Results//{version}'.format(version=self.version)
        self.svnm = '{nm}-{nele}-{temp}'.format(
            nm=self.version,
            nele=self.num_element_filter,
            temp='{}C'.format(self.temperature_filter) if self.temperature_filter is not None else 'All'
        )

        if not os.path.exists(self.svfl):
            os.makedirs(self.svfl)
            os.makedirs('{}\\{}'.format(self.svfl, 'trees'))
            os.makedirs('{}\\{}'.format(self.svfl, 'figures'))
            os.makedirs('{}\\{}'.format(self.svfl, 'htmls'))
            os.makedirs('{}\\{}'.format(self.svfl, 'features'))
            os.makedirs('{}\\{}'.format(self.svfl, 'eval'))

    def set_learner(self, learner, tuning=False, params='default'):
        """ Select which ML algorithm the learner should use.
          If tuning is True, then select parameter grid.
          ELse, selects appropriate parameters for ML learner. """

        learn_selector = {
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
        }

        if tuning:
            tuning_parameters = {
                'rfr': {
                    'n_estimators':          [10, 25, 50, 100, 200],
                    'max_features':          ['auto', 'sqrt'],
                    'max_depth':             [None, 3, 5, 10],
                    'min_samples_split':     [2, 5, 10],
                    'min_samples_leaf':      [1, 2, 3, 5],
                    'max_leaf_nodes':        [None, 5, 20, 50],
                    'min_impurity_decrease': [0, 0.1, 0.4]
                },

                'adaboost': {
                    'base_estimator': [None, tree.ExtraTreeRegressor()],
                    'n_estimators': [10,50,200,500],
                    'learning_rate': [0.5, 1, 2],
                    'loss': ['linear','square','exponential']
                },

                'tree': {
                    'criterion': ['mse', 'mae'],
                    'splitter': ['best','random'],
                    'max_depth': [None, 2, 5],
                    'min_samples_split': [2, 5, 0.1, 0.5],
                    'max_features': ['auto', 'sqrt']
                },

                'SGD': {
                    'loss': ['squared loss', 'huber', 'epsilon_insensitive'],
                    'penalty': ['none','l2','l1','elasticnet'],
                    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                    'learning_rate': ['optimal', 'invscaling', 'adaptive'],
                    'eta0': [1e-2, 1e-1, 1],
                    'power_t': [0.05, 0.5, 1.5]
                },

                'neuralnet': {
                    'hidden_layer_sizes': [1, 2, 3, 5, 10],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs'],
                    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': [1e-4, 1e-3, 1e-2],
                    # 'power_t': [0.05, 0.5, 1.5],
                    'max_iter': [100, 200, 500],
                    'tol': [1e-5, 1e-4, 1e-3],
                    'momentum': [0.8, 0.9, 0.95, 0.99],
                    'early_stopping': [True],
                    # 'n_iter_no_change': [10, 20, 50]
                },

                'svr': {
                    'epsilon': [1e-1, 1e-2, 1e-3],
                    'kernel':  ['linear', 'poly', 'rbf'],
                    'gamma':   [1, 1e-1, 1e-2, 'auto'],
                    # 'degree':  [2, 3, 5], # full dat set
                    'degree': [2, 3], # 3 cat set
                    'coef0':   [0, 1, 1e-1, 1e1, 1e2, 1e-2],
                    'max_iter': [200]
                },

                'knnr': {
                    # 'n_neighbors': [2, 5, 7, 10], # full dat set
                    'n_neighbors': [1, 2, 3, 4], # 3 cat set
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [2, 5, 10, 30, 50],
                    'p': [1, 2, 3],
                },

                'krr': {
                    'alpha': [0.9, 1, 1.1, 1.5],
                    'degree': [2, 3, 5],
                    'coef0': [0, 1, 5],
                },

                'etr': {
                    'n_estimators':          [10, 25, 50, 100, 200, 400],
                    'criterion':             ['mae'],
                    'max_features':          ['auto', 'sqrt', 'log2', 0.2, 0.1, 0.05, 0.01],
                    'max_depth':             [None, 3, 5, 10],
                    'min_samples_split':     [2, 5, 10],
                    'min_samples_leaf':      [1, 2, 3, 5],
                    'max_leaf_nodes':        [None, 5, 20, 50],
                    'min_impurity_decrease': [0, 0.1, 0.4]
                },

                'gbr': {
                    'loss':                  ['ls', 'lad', 'quantile', 'huber'],
                    'learning_rate':         [0.05, 0.1, 0.2],
                    'subsample':             [0.5, 1],
                    'n_estimators':          [25, 100, 500],
                    'max_depth':             [None, 3, 5, 10],
                    'criterion':             ['friedman_mse', 'mae'],
                    'min_samples_split':     [2, 5, 10],
                    'min_samples_leaf':      [1, 2, 3, 5],
                    'max_features':          ['auto', 'sqrt'],
                    'max_leaf_nodes':        [None, 5, 20, 50],
                    'min_impurity_decrease': [0, 0.1, 0.4]
                 },

                'ridge': {
                    'alpha': [0.9, 1, 1.1, 1.5],
                    'solver': ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'],
                },

                'lasso': {
                    'alpha': [0.9, 1, 1.1, 1.5],
                    'fit_intercept': [True, False],
                    'normalize': [True, False],
                    'max_iter': [100, 200, 500, 1000],
                },
            }

            self.machina = learn_selector.get(learner, lambda: 'Error')()
            return tuning_parameters.get(learner, lambda : 'Error')

        elif isinstance(params, dict):
            self.machina = learn_selector.get(learner, lambda: 'Error')()
            self.machina.set_params(**params)

        else:

            param_selector = {
                'rfr': {'n_estimators':25, 'max_depth':10, 'max_leaf_nodes':50, 'min_samples_leaf':1,
                            'min_samples_split':2, 'max_features':'auto', 'bootstrap':True, 'n_jobs':4,
                            'criterion':'mae'},
                'etr': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_impurity_decrease': 0,
                        'max_leaf_nodes': 50, 'max_features': 'auto', 'max_depth': 10, 'criterion': 'mae'},
                'etr-uncertainty': {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 4, 'min_impurity_decrease': 0,
                        'max_leaf_nodes': 50, 'max_features': 'auto', 'max_depth': 10, 'criterion': 'mae'},
                'etr-CaMnIn': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_impurity_decrease': 0,
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

            self.machina = learn_selector.get(learner, lambda: 'Error')()
            self.machina.set_params(**param_selector.get(params))

    def set_filters(self, element_filter=None, temperature_filter=None, ammonia_filter=None, space_vel_filter=None,
                 ru_filter=None, pressure_filter=None, promoter_filter=None):
        """ Update filters and reset naming convention """

        if element_filter is not None:
            self.num_element_filter = element_filter
        if temperature_filter is not None:
            self.temperature_filter = temperature_filter
        if ammonia_filter is not None:
            self.ammonia_filter = ammonia_filter
        if ru_filter is not None:
            self.ru_filter = ru_filter
        if pressure_filter is not None:
            self.pressure_filter = pressure_filter
        if space_vel_filter is not None:
            self.sv_filter = space_vel_filter
        if promoter_filter is not None:
            self.promoter_filter = promoter_filter

        self.set_name_paths()

    def reset_filters(self):
        """ Set all filter variables to None and reset naming convention """

        self.num_element_filter = None
        self.temperature_filter = None
        self.ammonia_filter = None
        self.ru_filter = None
        self.pressure_filter = None
        self.promoter_filter = None
        self.sv_filter = None

        self.set_name_paths()

    def set_temperature_filter(self, T):
        self.temperature_filter = T
        self.set_name_paths()

    def load_static_dataset(self, catalyst_container):
        """ Handoff from catalyst container to supervised learner """
        self.static_dataset = catalyst_container.master_container
        self.dynamic_dataset = self.static_dataset.copy()

    def filter_static_dataset(self, reset_training_data=True,  shuffle_dataset=True):
        """ Apply all filters to the dataset
        :param reset_training_data: overwrite training dataframes and variables with new values
        :param shuffle_dataset: randomize the order of data within the dynamic dataframe
        """

        self.reset_dynamic_dataset()
        self.filter_temperatures()
        self.filter_n_elements()
        self.filter_pressure()
        self.filter_concentrations()
        self.filter_ruthenium_loading()
        self.filter_space_velocities()
        self.filter_promoter()

        if shuffle_dataset:
            self.shuffle_dynamic_dataset()

        if reset_training_data:
            self.set_training_data() # This does nothing right now...

    def set_target_columns(self, cols):
        """ Define measured values for ML algorithm (target values) """

        if isinstance(cols, list):
            self.target_columns = cols
        else:
            self.target_columns = list(cols)

    def set_group_columns(self, cols):
        """ Define a group column for cross-validation models """

        if isinstance(cols, list):
            self.group_columns = cols
        else:
            self.group_columns = list(cols)

    def set_hold_columns(self, cols):
        """ Define hold columns - these are informational columns that are excluded from the feature set. """

        if isinstance(cols, list):
            self.hold_columns = cols
        else:
            self.hold_columns = list(cols)

    def set_drop_columns(self, cols):
        """ Define a drop column - these columns are permenantly removed. """
        if isinstance(cols, list):
            self.drop_columns = cols
        else:
            self.drop_columns = list(cols)

    def set_training_data(self):
        """ Use all specified columns to sort data into correct dataframes """

        self.features_df = self.dynamic_dataset.drop(
            labels=self.target_columns + self.group_columns + self.hold_columns, axis=1)
        self.drop_features()

        self.labels_df = self.dynamic_dataset[self.target_columns].copy()
        self.labels = self.labels_df.values
        if self.labels.shape[1] == 1:
            self.labels = np.ravel(self.labels)

        self.groups = self.dynamic_dataset[self.group_columns].values
        if self.groups.shape[1] == 1:
            self.groups = np.ravel(self.groups)

        self.hold_df = self.dynamic_dataset[self.hold_columns].copy()

    def reset_dynamic_dataset(self):
        """ Copy static dataset onto dynamic to refresh modified data. """

        self.dynamic_dataset = self.static_dataset.copy()

    def filter_n_elements(self):
        """ Filter by number of elements in the catalyst. """

        filter_dict_neles = {
            1: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 1],
            2: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 2],
            3: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 3],
            23: self.dynamic_dataset[(self.dynamic_dataset['n_elements'] == 2) |
                                     (self.dynamic_dataset['n_elements'] == 3)],
        }

        self.dynamic_dataset = filter_dict_neles.get(self.num_element_filter, self.dynamic_dataset)

    def filter_temperatures(self):
        """ Filter by temperature of catalyst observation. """

        if self.temperature_filter is None:
            self.dynamic_dataset = self.dynamic_dataset[self.dynamic_dataset.loc[:, 'temperature'] != 150]
        elif isinstance(self.temperature_filter, str):
            temp_dict = {
                'not450': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'temperature'] != 450) &
                                               (self.dynamic_dataset.loc[:, 'temperature'] != 150)],
                'not400': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'temperature'] != 400) &
                                               (self.dynamic_dataset.loc[:, 'temperature'] != 150)],
                'not350': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'temperature'] != 350) &
                                               (self.dynamic_dataset.loc[:, 'temperature'] != 150)],
                '350orless': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'temperature'] != 450) &
                                                  (self.dynamic_dataset.loc[:, 'temperature'] != 400) &
                                                  (self.dynamic_dataset.loc[:, 'temperature'] != 150)],
                '300orless': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'temperature'] != 450) &
                                                  (self.dynamic_dataset.loc[:, 'temperature'] != 400) &
                                                  (self.dynamic_dataset.loc[:, 'temperature'] != 350) &
                                                  (self.dynamic_dataset.loc[:, 'temperature'] != 150)],
                None: self.dynamic_dataset[self.dynamic_dataset.loc[:, 'temperature'] != 150]
            }

            self.dynamic_dataset = temp_dict.get(self.temperature_filter)
        else:
            self.dynamic_dataset = self.dynamic_dataset[self.dynamic_dataset.loc[:, 'temperature'] == self.temperature_filter]

    def filter_concentrations(self):
        """ Filter by ammonia concentration. """

        filter_dict_ammonia = {
            1: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'ammonia_concentration'] > 0.5) &
                                    (self.dynamic_dataset.loc[:, 'ammonia_concentration'] < 1.9)],
            5: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'ammonia_concentration'] > 4.8) &
                                    (self.dynamic_dataset.loc[:, 'ammonia_concentration'] < 5.2)]
        }

        self.dynamic_dataset = filter_dict_ammonia.get(self.ammonia_filter, self.dynamic_dataset)

    def filter_space_velocities(self):
        """ Filter by measured space velocity. """

        filter_dict_sv = {
            2000: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'space_velocity'] > 1400) &
                                       (self.dynamic_dataset.loc[:, 'space_velocity'] < 3000)]
        }

        self.dynamic_dataset = filter_dict_sv.get(self.sv_filter, self.dynamic_dataset)

    def filter_ruthenium_loading(self):
        """ Filter by ruthenium weight loading.  This is specific to the ammonia project. """

        filter_dict_ruthenium = {
            1: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.01)],
            2: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.02)],
            3: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03)],
            32: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                     (self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.02)],
            31: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                     (self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.01)],
            21: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.02) |
                                     (self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.01)],
            '3+': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] >= 0.03)],
            'mol3': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.0252)],
        }

        self.dynamic_dataset = filter_dict_ruthenium.get(self.ru_filter, self.dynamic_dataset)

    def filter_pressure(self):
        """ Filter by reaction pressure. """

        pass

    def filter_promoter(self):

        filter_dict_promoter = {
            'K12': self.dynamic_dataset[self.dynamic_dataset.loc[:, 'K Loading'] == 0.12],
        }

        self.dynamic_dataset = filter_dict_promoter.get(self.promoter_filter, self.dynamic_dataset)

    def filter_out_elements(self, eles):
        """ Remove specified elements (eles) from the dataset. """

        if isinstance(eles, list):
            for ele in eles:
                self.dynamic_dataset.drop(
                    self.dynamic_dataset.loc[self.dynamic_dataset['{} Loading'.format(ele)] > 0].index,
                    inplace=True
                )
        else:
            self.dynamic_dataset.drop(columns=['{} Loading'.format(eles)], inplace=True)

        self.shuffle_dynamic_dataset()

    def filter_out_ids(self, ids):
        """ Filter catalysts from the dataset by their ID numbers. """

        if isinstance(ids, list):
            for catid in ids:
                self.dynamic_dataset = self.dynamic_dataset[self.dynamic_dataset['ID'] != catid]
        else:
            self.dynamic_dataset = self.dynamic_dataset.drop(index=ids)

        self.shuffle_dynamic_dataset()

    def shuffle_dynamic_dataset(self, sv=False):
        """ Randomize the order of the dynamic dataset. """

        self.dynamic_dataset = shuffle(self.dynamic_dataset)

        if sv:
            pd.DataFrame(self.dynamic_dataset).to_csv('..\\Dynamic_df.csv')

        # Set up training data and apply grouping
        self.set_training_data()

    def drop_features(self):
        """ Use self.drop_columns to remove columns from the features dataframe. """

        if self.drop_columns is not None:
            cols = self.features_df.columns
            feature_list = list()
            for col in cols:
                if (col.split('_')[0] in self.drop_columns) | (col in self.drop_columns):
                    feature_list += [col]

            self.features_df.drop(columns=feature_list, inplace=True)
            self.features = self.features_df.values
        else:
            self.features = self.features_df.values

    def reduce_features(self):
        """ Use a feature selection algorithm to drop features. """

        rfe = RFE(estimator=self.machina, n_features_to_select=20)
        rfe.fit(self.features, self.labels)
        self.features_df[:] = rfe.inverse_transform(self.features)
        self.features_df.to_csv('{}\\{}'.format(self.svfl, 'feature_list.csv'))

    def set_training_set(self, training_elements=None):
        pass # TODO I want to come up with a clever way to segment into training and test sets...

    def train_data(self):
        """ Train the model on feature/label datasets """

        self.machina = self.machina.fit(self.features, self.labels)

    def predict_data(self):
        """ Use a trained model to predict based on new features. """

        self.predictions = self.machina.predict(self.features)
        return self.predictions

    def predict_crossvalidate(self, kfold=None):
        """ Use k-fold validation with grouping by catalyst ID to determine. """

        if isinstance(kfold, int):
            if kfold > 1:
                self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                                     groups=self.groups, cv=GroupKFold(kfold))
            else:
                print('Invalid kfold. Resorting to 10-fold validation.')
                self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                                     groups=self.groups, cv=GroupKFold(10))
        elif kfold == 'LOO':
            self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                                 groups=self.groups, cv=LeaveOneGroupOut())
        elif kfold == 'LSO':
            self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                                 cv=LeaveOneOut())
        else:
            print('Invalid kfold. Resorting to 10-fold validation.')
            self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                                 groups=self.groups, cv=GroupKFold(10))

    def calculate_tau(self):
        """ Calculate tau, the uncertainty scale factor, from known data. Tau is used to estimate MAE in unknown
        parameter space.
        """

        tree_predition_df = pd.DataFrame(index=self.features_df.index)

        # Use each tree in the forest to generate the individual tree prediction
        for nth_tree, tree in enumerate(self.machina.estimators_):
            tree_predition_df.loc[:, 'Tree {}'.format(nth_tree)] = tree.predict(self.features)

        # Remove observations (i.e. trees) that are outside 90% CI (less than 5% or greater than 95%)
        forest_stats = tree_predition_df.apply(lambda x: np.percentile(a=x, q=[0, 100]), axis=1)
        for idx, rw in tree_predition_df.iterrows():
            forest_min = forest_stats[idx][0]
            forest_max = forest_stats[idx][1]
            rw[(rw > forest_max) | (rw < forest_min)] = np.nan
            tree_predition_df.loc[idx] = rw

        # Calculate scaling parameter per...
        # J. W. Coulston, C. E. Blinn, V. A. Thomas, R. H. Wynne,
        # Approximating Prediction Uncertainty for Random Forest Regression Models.
        # Photogramm. Eng. Remote Sens. 82, 189–197 (2016).
        tau_array = np.sqrt(
            (self.labels - tree_predition_df.mean(axis=1).values)**2 / tree_predition_df.var(axis=1).values
        )
        self.tau = np.nanmean(tau_array)
        print('Tau: {}'.format(self.tau))

    def calculate_uncertainty(self):
        """ Calculate the uncertainty of new predictions.  These predictions are scaled by Tau to estimate MAE. """

        tree_predition_df = pd.DataFrame(index=self.features_df.index)

        # Use each tree in the forest to generate the individual tree prediction
        for nth_tree, tree in enumerate(self.machina.estimators_):
            tree_predition_df.loc[:, 'Tree {}'.format(nth_tree)] = tree.predict(self.features)

        # print(tree_predition_df.var(axis=1))
        self.uncertainty = np.sqrt(self.tau**2 * tree_predition_df.var(axis=1).values)

    def calculate_bias(self):
        #TODO: Finish bias
        n_samples = len(self.labels)
        sq_bias = 1/n_samples * (np.mean(self.labels) - self.labels)**2
        print(sq_bias)

    def calculate_variance(self):
        # TODO: Finish variance
        n_samples = len(self.labels)
        vari = 1 / n_samples * (np.mean(self.labels) - self.labels) ** 2
        print(vari)

    def predict_leave_one_out(self):
        print('Method predict_leave_one_out depricated: Change to predict_crossvalidate.')
        self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                             groups=self.groups, cv=LeaveOneGroupOut())

    def predict_leave_self_out(self):
        print('Method predict_leave_self_out depricated: Change to predict_crossvalidate.')
        self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                             cv=LeaveOneOut())

    def evaluate_regression_learner(self):
        """ Calculate model evaluation parameters, print, and save. """

        r2 = r2_score(self.labels_df.values, self.predictions)
        mean_abs_err = mean_absolute_error(self.labels_df.values, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.labels_df.values, self.predictions))

        print('\n----- Model {} -----'.format(self.svnm))
        print('R2: {:0.3f}'.format(r2))
        print('Mean Absolute Error: {:0.3f}'.format(mean_abs_err))
        print('Root Mean Squared Error: {:0.3f}'.format(rmse))
        print('Time to Complete: {:0.1f} s'.format(time.time() - self.start_time))
        print('\n')

        pd.DataFrame([r2, mean_abs_err, rmse, time.time() - self.start_time],
                     index=['R2','Mean Abs Error','Root Mean Squared Error','Time']
                     ).to_csv('{}\\eval\\{}-eval.csv'.format(self.svfl, self.svnm))

    def predict_all_from_elements(self, elements, loads=None, cv=False):
        """ Use given elements as a training dataset to predict all other catalysts

        :param elements: Elements included in training dataset
        :param loads: Loading of elements to be included
        :param cv: Type of cross validation to perform (default no CV, only make prediction)
        :return: results dataset
        """

        # Refresh dynamic dataset
        self.filter_static_dataset()

        # Create a dataframe of all element/load pairs to filter
        filter_df = pd.DataFrame([elements, loads], index=['Element', 'Loading']).T
        training_index_list = list()

        # Set training and test data index lists
        for idx, rw in filter_df.iterrows():
            training_index_list += self.dynamic_dataset[self.dynamic_dataset['{} Loading'.format(rw['Element'])] == rw['Loading']].index.values.tolist()

        dynamic_index_list = self.dynamic_dataset.index.values.tolist()
        test_data_index_list = list(set(dynamic_index_list) - set(training_index_list))

        # Drop test data from dataset
        self.dynamic_dataset.drop(index=test_data_index_list, inplace=True)
        self.set_training_data()

        # Train model
        self.train_data()

        if cv is not False:
            if cv is True:
                self.predict_crossvalidate(kfold='LSO')
            else:
                self.predict_crossvalidate(kfold=cv)
        else:
            # Refresh dynamic dataset
            self.filter_static_dataset()

            # Drop training data from dynamic dataset
            self.dynamic_dataset.drop(index=training_index_list, inplace=True)
            self.set_training_data()

            # Predict test data and compile results
            self.predict_data()

        self.compile_results(sv=False)

        return self.result_dataset

    def compile_results(self, sv=False, svnm=None):
        """ Create a results dataframe that merges dynamic with hold dataframe """

        # Create Result DF, add predictions and experimental data
        self.result_dataset = self.dynamic_dataset[self.features_df.columns].copy()

        """ Add predictions and labels. """
        try:
            self.result_dataset['Predicted Conversion'] = self.predictions
        except ValueError:
            print('No Predictions Generated by model...')

        self.result_dataset['Measured Conversion'] = self.labels

        """ Parse Catalyst Names """
        for index, edict in self.dynamic_dataset['Element Dictionary'].iteritems():
            self.result_dataset.loc[index, 'Name'] = ''.join('{}({})'.format(key, str(int(val))) for key, val in edict)

            i = 1
            for key, val in edict:
                self.result_dataset.loc[index, 'Ele{}'.format(i)] = key
                self.result_dataset.loc[index, 'Load{}'.format(i)] = val
                i += 1

        """ Add uncertainty. """
        try:
            self.result_dataset['Uncertainty'] = self.uncertainty
        except ValueError:
            print('No Uncertainty Generated')

        """ Save if requested. """
        if sv:
            if svnm is None:
                if self.svnm is not None:
                    self.result_dataset.to_csv('{}\\result_dataset-{}.csv'.format(self.svfl, self.svnm))
            else:
                self.result_dataset.to_csv('{}\\result_dataset-{}.csv'.format(self.svfl, svnm))

    def save_dynamic(self):
        """ Save RAW dynamic dataset - Use "compile_results" method unless debugging """
        self.dynamic_dataset.to_csv('{}\dynamic_data-{}.csv'.format(self.svfl, self.svnm))

    def extract_important_features(self, sv=False, prnt=False):
        """ Save all feature importance, print top 10 """

        try:
            feature_importance_df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns,
                          columns=['Feature Importance'])
        except AttributeError:
            return

        if prnt:
            print(feature_importance_df.sort_values(by='Feature Importance', ascending=False).head(10))

        if sv:
            feature_importance_df.to_csv('{}//features//feature_importance-{}.csv'.format(self.svfl, self.svnm))

            new_df = pd.DataFrame()

            for nm in feature_importance_df.index:
                feature_importance_df.loc[nm, 'Feature'] = nm.split('_')[0]

            for feat in feature_importance_df.Feature.unique():
                new_df.loc[feat, 'Feature Importance'] = feature_importance_df[feature_importance_df['Feature'] == feat]['Feature Importance'].sum()

            new_df.sort_values('Feature Importance', ascending=False, inplace=True)
            new_df.to_csv('{}//features//feature_importance-{}-summed.csv'.format(self.svfl, self.svnm))

        return feature_importance_df

    def hyperparameter_tuning(self, grid=False):
        """ Method Used to tune hyperparameters and increase accuracy of the model """
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

        svm_tuning_params = {
            'epsilon': [1, 1e-1, 1e-2, 1e-3, 0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'gamma': [1, 1e-1, 1e-2, 'auto'],
            'degree': [2, 3, 5, 7],
            'coef0': [0, 1, 1e-1, 1e1, 1e2, 1e-2]
        }

        self.machina_tuning_parameters = svm_tuning_params

        if grid:
            gs = GridSearchCV(self.machina, self.machina_tuning_parameters, cv=3, return_train_score=True)
        else:
            gs = RandomizedSearchCV(self.machina, self.machina_tuning_parameters, cv=GroupKFold(3),
                                    return_train_score=True, n_iter=200)

        gs.fit(X=self.features, y=self.labels, groups=self.groups)
        pd.DataFrame(gs.cv_results_).to_csv('{}\\p-tune-svm_{}.csv'.format(self.svfl, self.svnm))

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

    def generate_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.machina,
            X=self.features_df.values,
            y=self.labels_df.values,
            groups=self.groups,
            scoring=make_scorer(score_func=mean_absolute_error, greater_is_better=True),
            train_sizes=np.linspace(0.05,1.0,20),
            cv=GroupKFold(10),
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.grid()

        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Absolute Error")

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
        #          label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.ylim(0, 0.4)

        plt.legend(loc="best")
        plt.show()

    def save_model_parameters_to_csv(self):
        """ Save all model filter parameters to csv file. """

        pd.DataFrame(
            [
                self.num_element_filter,
                self.temperature_filter,
                self.ammonia_filter,
                self.ru_filter,
                self.pressure_filter,
                self.sv_filter,
                self.version,
                self.target_columns,
                self.drop_columns,
                self.group_columns,
                self.hold_columns,

            ]
        ).to_csv('{}//eval//{}_modelparam.csv'.format(self.svfl, self.svnm))

    def find_optimal_feature_count(self):
        ''' '''
        # TODO groups is not working for this method for some reason...

        rfe = RFECV(estimator=self.machina, cv=GroupKFold(10), scoring='neg_mean_squared_error')
        rfe.fit(X=self.features, y=self.labels)
        # rfe.fit(X=self.features, y=self.labels, groups=self.groups)

        print("Optimal number of features : %d" % rfe.n_features_)

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
        plt.show()

    def random_feature_test(self, combined=False):
        ''' Create a set of random features to test feature efficacy '''

        random_features = np.random.random_sample(self.features.shape)

        if combined is True:
            rand_df = pd.DataFrame(
                random_features,
                columns=['RandFeat {}'.format(i) for i in range(len(self.features.transpose()))],
                index=self.features_df.index
            )

            self.features_df = pd.concat([self.features_df, rand_df], axis=1)

        elif combined is 'temp_only':
            # temperature is correct, but everything else is random
            self.features_df[:] = random_features
            self.features_df.columns = ['RandFeat {}'.format(i) for i in range(len(self.features.transpose()) - 1)] + [
                'temperature']
            self.features_df['temperature'] = self.dynamic_dataset['temperature']

        elif combined is 'temp_and_weights':
            self.features_df[:] = random_features
            self.features_df.columns = ['RandFeat {}'.format(i) for i in range(len(self.features.transpose()) - 1)] + [
                'temperature']

            self.features_df['temperature'] = self.dynamic_dataset['temperature']

            weight_loading_columns = [col for col in self.dynamic_dataset.columns if 'Loading' in col]
            self.features_df = pd.concat([self.features_df, self.dynamic_dataset.loc[:, weight_loading_columns]], axis=1)

        else:
            self.features_df[:] = random_features
            self.features_df.columns = ['RandFeat {}'.format(i) for i in range(len(self.features.transpose())-1)] + ['temperature']
            rand_temperature = np.random.choice([250, 300, 350], len(self.features))
            self.features_df['temperature'] = rand_temperature

        self.dynamic_dataset = pd.concat(
            [self.features_df,
             self.labels_df,
             self.hold_df,
             pd.DataFrame(self.groups, index=self.labels_df.index)
             ],
            axis=1
        )

        self.features = self.features_df.values

    def drop_all_features(self, exclude=None):
        if exclude is None:
            self.features_df = pd.DataFrame()
        else:
            self.features_df.drop(columns=[x for x in self.features_df.columns if x not in exclude], inplace=True)

        self.dynamic_dataset = pd.concat(
            [self.features_df,
             self.labels_df,
             self.hold_df,
             pd.DataFrame(self.groups, index=self.labels_df.index)
             ],
            axis=1
        )

        self.features = self.features_df.values

