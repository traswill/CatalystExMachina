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
            for ele, wt in catobj.elements.items():
                load_df.loc[catid, '{} Loading'.format(ele)] = wt / 100

            # Create group
            groupdf = pd.DataFrame(catobj.group, index=[catid], columns=['group'])

            # Create DF from features
            featdf = pd.DataFrame.from_dict(catobj.feature_dict, orient='index').transpose()
            featdf.index = [catid]

            # Create element dictionary
            eldictdf = pd.DataFrame(catobj.elements.items(), index=[catid], columns=['Element Dictionary'])

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

    def __init__(self, version='v00'):
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

        self.hold_df = pd.DataFrame()
        self.features_df = pd.DataFrame()
        self.labels_df = pd.DataFrame()

        self.features = np.empty(1)
        self.labels = np.empty(1)
        self.groups = np.empty(1)
        self.predictions = list()

        self.features_to_drop = None

        # '''Initialize DataFrames for unchanging data (master) and sorting/filtering (slave)'''
        # self.static_dataset = pd.DataFrame()  # From Catalyst Container
        # self.dynamic_dataset = pd.DataFrame()
        # self.tester_dataset = pd.DataFrame()
        # self.features_to_drop = None
        #
        # '''Initialize sub-functions from the worker dataset.'''
        # self.features_df = pd.DataFrame()
        # self.labels_df = pd.DataFrame()
        # self.plot_df = pd.DataFrame()
        # self.feature_importance_df = pd.DataFrame()
        # self.predictions = list()
        # self.groups = None

        '''Initialize ML algorithm'''
        self.machina = None

        '''Initialize all options for the algorithm.  These are used in naming files.'''
        self.num_element_filter = 0
        self.temperature_filter = None
        self.ammonia_filter = None
        self.ru_filter = None
        self.pressure_filter = None
        self.sv_filter = None
        self.version = version

        '''Create path, folder, and filename'''
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

        '''Initialize Time for run-length statistics'''
        self.start_time = time.time()

    def set_name_paths(self):
        """ These paths are used by all methods to save files to the proper location """

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

    def set_learner(self, learner, params='default'):
        """ Select which ML algorithm the learner should use.  Also selects appropriate parameters. """
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

        param_selector = {
            'rfr': {'n_estimators':25, 'max_depth':10, 'max_leaf_nodes':50, 'min_samples_leaf':1,
                        'min_samples_split':2, 'max_features':'auto', 'bootstrap':True, 'n_jobs':4,
                        'criterion':'mae'},
            'etr': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_impurity_decrease': 0,
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
                 ru_filter=None, pressure_filter=None):

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

        self.set_name_paths()

    def reset_filters(self):
        self.num_element_filter = None
        self.temperature_filter = None
        self.ammonia_filter = None
        self.ru_filter = None
        self.pressure_filter = None
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
        """ Apply all filters to the dataset """
        self.reset_dynamic_dataset()
        self.filter_temperatures()
        self.filter_n_elements()
        self.filter_pressure()
        self.filter_concentrations()
        self.filter_ruthenium_loading()
        self.filter_space_velocities()

        if shuffle_dataset:
            self.shuffle_dynamic_dataset()

        if reset_training_data:
            self.set_training_data()

    def set_target_columns(self, cols):
        if isinstance(cols, list):
            self.target_columns = cols
        else:
            self.target_columns = list(cols)

    def set_group_columns(self, cols):
        if isinstance(cols, list):
            self.group_columns = cols
        else:
            self.group_columns = list(cols)

    def set_hold_columns(self, cols):
        if isinstance(cols, list):
            self.hold_columns = cols
        else:
            self.hold_columns = list(cols)

    def set_drop_columns(self, cols):
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
        self.dynamic_dataset = self.static_dataset.copy()

    def filter_n_elements(self):
        filter_dict_neles = {
            1: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 1],
            2: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 2],
            3: self.dynamic_dataset[self.dynamic_dataset['n_elements'] == 3],
            23: self.dynamic_dataset[(self.dynamic_dataset['n_elements'] == 2) |
                                     (self.dynamic_dataset['n_elements'] == 3)],
        }

        self.dynamic_dataset = filter_dict_neles.get(self.num_element_filter, self.dynamic_dataset)

    def filter_temperatures(self):
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
        filter_dict_ammonia = {
            1: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'ammonia_concentration'] > 0.5) &
                                    (self.dynamic_dataset.loc[:, 'ammonia_concentration'] < 1.5)],
            5: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'ammonia_concentration'] > 4.8) &
                                    (self.dynamic_dataset.loc[:, 'ammonia_concentration'] < 5.2)]
        }

        self.dynamic_dataset = filter_dict_ammonia.get(self.ammonia_filter, self.dynamic_dataset)

    def filter_space_velocities(self):
        filter_dict_sv = {
            2000: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'space_velocity'] > 1400) &
                                       (self.dynamic_dataset.loc[:, 'space_velocity'] < 2600)]
        }

        self.dynamic_dataset = filter_dict_sv.get(self.sv_filter, self.dynamic_dataset)

    def filter_ruthenium_loading(self):
        filter_dict_ruthenium = {
            1: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.01)],
            2: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.02)],
            3: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03)],
            32: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                     (self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.02)],
            31: self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                     (self.dynamic_dataset.loc[:, 'Ru Loading'] == 0.01)],
            '3+': self.dynamic_dataset[(self.dynamic_dataset.loc[:, 'Ru Loading'] >= 0.03)],
        }

        self.dynamic_dataset = filter_dict_ruthenium.get(self.ru_filter, self.dynamic_dataset)

    def filter_pressure(self):
        pass

    def filter_out_elements(self, eles):
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
        if isinstance(ids, list):
            for catid in ids:
                self.dynamic_dataset = self.dynamic_dataset[self.dynamic_dataset['ID'] != catid]
        else:
            self.dynamic_dataset = self.dynamic_dataset.drop(index=ids)

        self.shuffle_dynamic_dataset()

    def shuffle_dynamic_dataset(self, sv=False):
        self.dynamic_dataset = shuffle(self.dynamic_dataset)

        if sv:
            pd.DataFrame(self.dynamic_dataset).to_csv('..\\Dynamic_df.csv')

        # Set up training data and apply grouping
        self.set_training_data()
    #     self.group_for_training()
    #
    # def group_for_training(self):
    #     """ Set groups parameter AFTER shuffling the slave dataset """
    #     self.groups = self.dynamic_dataset[self.group_columns].values

    # TODO this method should be offloaded to an operator
    def reduce_feature_set(self):
        reduced_features = [
            'temperature', 'Number d-shell Valence Electrons_mean', 'Number d-shell Valence Electrons_mad'
        ]
        self.drop_columns = [x for x in list(self.dynamic_dataset.columns) if x not in reduced_features]

    def drop_features(self):
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

    def set_training_set(self, training_elements=None):
        pass # TODO I want to come up with a clever way to segment into training and test sets...

    def train_data(self):
        """ Train the model on feature/label datasets """
        self.machina = self.machina.fit(self.features, self.labels)

    def predict_data(self):
        self.predictions = self.machina.predict(self.features)

    def predict_crossvalidate(self, kfold=10):
        """ Use k-fold validation with grouping by catalyst ID to determine  """
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

    def predict_leave_one_out(self):
        print('Method predict_leave_one_out depricated: Change to predict_crossvalidate.')
        self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                             groups=self.groups, cv=LeaveOneGroupOut())

    def predict_leave_self_out(self):
        print('Method predict_leave_self_out depricated: Change to predict_crossvalidate.')
        self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                             cv=LeaveOneOut())

    def evaluate_regression_learner(self):
        """ Comment """
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

    def predict_within_elements(self, elements, svnm='data'):
        # TODO rework this method
        element_dataframe = pd.DataFrame()

        for ele in elements:
            dat = self.dynamic_dataset.loc[self.dynamic_dataset['{} Loading'.format(ele)] > 0]
            element_dataframe = pd.concat([element_dataframe, dat])

        self.labels_df = element_dataframe.loc[:, 'Measured Conversion'].copy()

        self.features_df = element_dataframe.drop(
            labels=['Measured Conversion', 'Element Dictionary', 'group'],
            axis=1
        )

        self.groups = element_dataframe['group'].values

        self.train_data()
        self.predict_crossvalidate(kfold=3)

        predvals = self.predictions
        measvals = self.labels_df.values

        comparison_df = pd.DataFrame([predvals, measvals],
                                     index=['Predicted Conversion', 'Measured Conversion'],
                                     columns=self.features_df.index).T

        comparison_df['ID'] = comparison_df.index
        comparison_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in element_dataframe['Element Dictionary']
        ]
        comparison_df['temperature'] = self.features_df['temperature']

        feat_df = self.extract_important_features()
        feat_df.to_csv('{}\\{}-features.csv'.format(self.svfl, svnm))

        g = Graphic(learner=self, df=comparison_df)
        g.plot_err(svnm='{}_predict-self_{}'.format(self.version, svnm))
        g.plot_err(svnm='{}_predict-self_{}_nometa'.format(self.version, svnm), metadata=False)
        g.plot_important_features(svnm=svnm)
        g.bokeh_predictions(svnm='{}_predict-self_{}'.format(self.version, svnm))

    def predict_all_from_elements(self, elements, loads=None, svnm='data', save_plots=True, save_features=True):
        # TODO rework this method
        element_dataframe = pd.DataFrame()

        for jj, ele in enumerate(elements):
            if loads is None:
                dat = self.dynamic_dataset.loc[self.dynamic_dataset['{} Loading'.format(ele)] > 0]
            else:
                dat = self.dynamic_dataset.loc[self.dynamic_dataset['{} Loading'.format(ele)] == loads[jj]]

            element_dataframe = pd.concat([element_dataframe, dat])

        drop_ids = np.unique(element_dataframe.index)

        self.labels_df = element_dataframe.loc[:, 'Measured Conversion'].copy()

        self.features_df = element_dataframe.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'group'],
                axis=1
            )

        self.train_data()

        test_data = self.dynamic_dataset.drop(index=drop_ids).copy()

        predvals = self.machina.predict(test_data.drop(labels=['Measured Conversion', 'Element Dictionary', 'group'],
            axis=1))

        measvals = test_data.loc[:, 'Measured Conversion'].values

        comparison_df = pd.DataFrame([predvals, measvals],
                                     index=['Predicted Conversion', 'Measured Conversion'],
                                     columns=test_data.index).T

        comparison_df['ID'] = comparison_df.index  # [x.split('_')[0] for x in comparison_df.index]
        comparison_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in test_data['Element Dictionary']
        ]
        comparison_df['temperature'] = test_data['temperature']

        if save_features:
            feat_df = self.extract_important_features()
            feat_df.to_csv('{}\\{}-features.csv'.format(self.svfl, svnm))

        if save_plots:
            g = Graphic(learner=self, df=comparison_df)
            g.plot_err(svnm='{}-predict_{}'.format(self.version, svnm))
            g.plot_err(svnm='{}-predict_{}_nometa'.format(self.version, svnm), metadata=False)
            g.plot_important_features(svnm=svnm)
            g.bokeh_predictions(svnm='{}-predict_{}'.format(self.version, svnm))

        return comparison_df

    def predict_from_master_dataset(self):
        """ Description """
        # Note: This may break due to significant changes in the learner methods

        data = self.static_dataset[self.static_dataset.index.str.contains('Predict') == True]
        data = data.drop(labels=['Measured Conversion', 'Element Dictionary', 'group'], axis=1)
        predvals = self.machina.predict(data.values)
        data['Predictions'] = predvals
        data.to_csv(r'{}/{}-BinaryPredictions.csv'.format(self.svfl, self.version))
        return data

    def compile_results(self, svnm=None):
        """ Prepare all data for plotting """

        if self.predictions is None:
            print('WARNING: No predictions have been made.')

        # Create Result DF and add Predictions
        self.result_dataset = self.dynamic_dataset[self.features_df.columns].copy()
        self.result_dataset['Predicted Conversion'] = self.predictions
        self.result_dataset['Measured Conversion'] = self.labels

        # Parse the Element Dictionary
        for index, edict in self.dynamic_dataset['Element Dictionary'].iteritems():
            self.result_dataset.loc[index, 'Name'] = ''.join('{}({})'.format(key, str(int(val))) for key, val in edict)

            i = 1
            for key, val in edict:
                self.result_dataset.loc[index, 'Ele{}'.format(i)] = key
                self.result_dataset.loc[index, 'Load{}'.format(i)] = val
                i += 1

        # Save Results and Features
        if svnm is None:
            self.result_dataset.to_csv('{}\\result_dataset-{}.csv'.format(self.svfl, self.svnm))
        else:
            self.result_dataset.to_csv('{}\\result_dataset-{}.csv'.format(self.svfl, svnm))

    def save_dynamic(self):
        """ Comment """
        self.dynamic_dataset.to_csv('{}\slavedata-{}.csv'.format(self.svfl, self.svnm))

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

        if grid:
            gs = GridSearchCV(self.machina, self.machina_tuning_parameters, cv=10, return_train_score=True)
        else:
            gs = RandomizedSearchCV(self.machina, etr_tuning_params, cv=GroupKFold(3),
                                return_train_score=True, n_iter=1000)

        gs.fit(self.features_df.values, self.labels_df.values, groups=self.groups)
        pd.DataFrame(gs.cv_results_).to_csv('{}\\p-tune-gbr_{}.csv'.format(self.svfl, self.svnm))

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
        df = self.static_dataset[['Element Dictionary']].copy()
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

