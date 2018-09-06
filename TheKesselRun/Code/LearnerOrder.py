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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, GroupKFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

            # Create DF from activity
            # actdf = pd.DataFrame(catobj.activity, index=[catid], columns=['Selectivity', 'Measured Conversion'])

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


class SupervisedLearner():
    """SupervisedLearner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self, version='v00'):
        """ Put Words Here """

        '''Initialize dictionary to hold import data'''
        self.catalyst_dictionary = dict()  # TODO Remove

        '''Initialize DataFrames for unchanging data (master) and sorting/filtering (slave)'''
        self.master_dataset = pd.DataFrame()  # From Catalyst Container
        self.slave_dataset = pd.DataFrame()
        self.tester_dataset = pd.DataFrame()
        self.features_to_drop = None

        '''Initialize sub-functions from the slave dataset.'''
        self.features_df = pd.DataFrame()
        self.labels_df = pd.DataFrame()
        self.plot_df = pd.DataFrame()
        self.feature_importance_df = pd.DataFrame()
        self.predictions = list()
        self.groups = None

        self.test_df = pd.DataFrame()
        self.train_df = pd.DataFrame()

        '''Initialize ML algorithm and tuning parameters'''
        self.machina = None

        '''Initialize all options for the algorithm.  These are used in naming files.'''
        self.element_filter = 0
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
            nele=self.element_filter,
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
        self.svfl = '..//Results//{version}'.format(version=self.version)
        self.svnm = '{nm}-{nele}-{temp}'.format(
            nm=self.version,
            nele=self.element_filter,
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
        # TODO add "if...not None, do" syntax
        self.element_filter = element_filter
        self.temperature_filter = temperature_filter
        self.ammonia_filter = ammonia_filter
        self.ru_filter = ru_filter
        self.pressure_filter = pressure_filter
        self.sv_filter = space_vel_filter

        self.set_name_paths()

    def set_temperature_filter(self, T):
        self.temperature_filter = T
        self.set_name_paths()

    def load_master_dataset(self, catalyst_container):
        self.master_dataset = catalyst_container.master_container
        self.slave_dataset = self.master_dataset.copy()

    def filter_master_dataset(self):
        self.reset_slave_dataset()
        self.filter_temperatures()
        self.filter_n_elements()
        self.filter_pressure()
        self.filter_concentrations()
        self.filter_ruthenium_loading()
        self.filter_space_velocities()
        self.drop_features()
        self.shuffle_slave()

    def set_training_data(self):
        ''' Use the slave dataframe to set other dataframe properties '''
        self.features_df = self.slave_dataset.drop(labels=['Measured Conversion', 'Element Dictionary', 'group'], axis=1)
        self.labels_df = self.slave_dataset['Measured Conversion'].copy()

    def reset_slave_dataset(self):
        self.slave_dataset = self.master_dataset.copy()

    def filter_n_elements(self):
        filter_dict_neles = {
            1: self.slave_dataset[self.slave_dataset['n_elements'] == 1],
            2: self.slave_dataset[self.slave_dataset['n_elements'] == 2],
            3: self.slave_dataset[self.slave_dataset['n_elements'] == 3],
            23: self.slave_dataset[(self.slave_dataset['n_elements'] == 2) |
                                    (self.slave_dataset['n_elements'] == 3)],
        }

        self.slave_dataset = filter_dict_neles.get(self.element_filter, self.slave_dataset)

    def filter_temperatures(self):
        if self.temperature_filter is None:
            self.slave_dataset = self.slave_dataset[self.slave_dataset.loc[:, 'temperature'] != 150]
        elif isinstance(self.temperature_filter, str):
            temp_dict = {
                'not450': self.slave_dataset[(self.slave_dataset.loc[:, 'temperature'] != 450) &
                                              (self.slave_dataset.loc[:, 'temperature'] != 150)],
                'not400': self.slave_dataset[(self.slave_dataset.loc[:, 'temperature'] != 400) &
                                              (self.slave_dataset.loc[:, 'temperature'] != 150)],
                'not350': self.slave_dataset[(self.slave_dataset.loc[:, 'temperature'] != 350) &
                                              (self.slave_dataset.loc[:, 'temperature'] != 150)],
                '350orless': self.slave_dataset[(self.slave_dataset.loc[:, 'temperature'] != 450) &
                                                 (self.slave_dataset.loc[:, 'temperature'] != 400) &
                                                 (self.slave_dataset.loc[:, 'temperature'] != 150)],
                '300orless': self.slave_dataset[(self.slave_dataset.loc[:, 'temperature'] != 450) &
                                                 (self.slave_dataset.loc[:, 'temperature'] != 400) &
                                                 (self.slave_dataset.loc[:, 'temperature'] != 350) &
                                                 (self.slave_dataset.loc[:, 'temperature'] != 150)],
                None: self.slave_dataset[self.slave_dataset.loc[:, 'temperature'] != 150]
            }

            self.slave_dataset = temp_dict.get(self.temperature_filter)
        else:
            self.slave_dataset = self.slave_dataset[self.slave_dataset.loc[:, 'temperature'] == self.temperature_filter]

    def filter_concentrations(self):
        filter_dict_ammonia = {
            1: self.slave_dataset[(self.slave_dataset.loc[:, 'ammonia_concentration'] > 0.5) &
                                   (self.slave_dataset.loc[:, 'ammonia_concentration'] < 1.5)],
            5: self.slave_dataset[(self.slave_dataset.loc[:, 'ammonia_concentration'] > 4.8) &
                                   (self.slave_dataset.loc[:, 'ammonia_concentration'] < 5.2)]
        }

        self.slave_dataset = filter_dict_ammonia.get(self.ammonia_filter, self.slave_dataset)

    def filter_space_velocities(self):
        filter_dict_sv = {
            2000: self.slave_dataset[(self.slave_dataset.loc[:, 'space_velocity'] > 1400) &
                                      (self.slave_dataset.loc[:, 'space_velocity'] < 2600)]
        }

        self.slave_dataset = filter_dict_sv.get(self.sv_filter, self.slave_dataset)

    def filter_ruthenium_loading(self):
        filter_dict_ruthenium = {
            1: self.slave_dataset[(self.slave_dataset.loc[:, 'Ru Loading'] == 0.01)],
            2: self.slave_dataset[(self.slave_dataset.loc[:, 'Ru Loading'] == 0.02)],
            3: self.slave_dataset[(self.slave_dataset.loc[:, 'Ru Loading'] == 0.03)],
            32: self.slave_dataset[(self.slave_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                   (self.slave_dataset.loc[:, 'Ru Loading'] == 0.02)],
            31: self.slave_dataset[(self.slave_dataset.loc[:, 'Ru Loading'] == 0.03) |
                                   (self.slave_dataset.loc[:, 'Ru Loading'] == 0.01)],
        }

        self.slave_dataset = filter_dict_ruthenium.get(self.ru_filter, self.slave_dataset)

    def filter_pressure(self):
        pass

    def filter_out_elements(self, eles):
        if isinstance(eles, list):
            for ele in eles:
                self.slave_dataset.drop(self.slave_dataset.loc[self.slave_dataset['{} Loading'.format(ele)] > 0].index,
                                        inplace=True)
        else:
            self.slave_dataset.drop(columns=['{} Loading'.format(eles)], inplace=True)

        self.shuffle_slave()

    def filter_out_ids(self, ids):
        if isinstance(ids, list):
            for catid in ids:
                self.slave_dataset = self.slave_dataset.drop(index=catid)
        else:
            self.slave_dataset = self.slave_dataset.drop(index=ids)

        self.shuffle_slave()

    def shuffle_slave(self, sv=False):
        self.slave_dataset = shuffle(self.slave_dataset)

        if sv:
            pd.DataFrame(self.slave_dataset).to_csv('..\\SlaveTest.csv')

        # Set up training data and apply grouping
        self.set_training_data()
        self.group_for_training()
        # self.trim_slave_dataset()

    def group_for_training(self):
        """ Set groups parameter AFTER shuffling the slave dataset """
        self.groups = self.slave_dataset['group'].values

    def reduce_feature_set(self):
        self.features_df = self.features_df.loc[
                           :, ['temperature',
                               'Number d-shell Valence Electrons_wt-mean', 'Number d-shell Valence Electrons_wt-mad',
                               ]
                           ].copy()

    def set_features_to_drop(self, features):
        self.features_to_drop = features

    def drop_features(self):
        if self.features_to_drop is not None:
            cols = self.slave_dataset.columns
            feature_list = list()
            for col in cols:
                if (col.split('_')[0] in self.features_to_drop) | (col in self.features_to_drop):
                    feature_list += [col]

            self.slave_dataset.drop(columns=feature_list, inplace=True)

    def set_training_set(self, training_elements=None):
        pass # I want to come up with a clever way to segment into training and test sets...

    def train_data(self):
        """ Train the model on feature/label datasets """
        self.machina = self.machina.fit(self.features_df.values, self.labels_df.values)

    def predict_data(self):
        self.predictions = self.machina.predict(self.features_df.values)

    def predict_crossvalidate(self, kfold=10, add_to_slave=False):
        """ Use k-fold validation with grouping by catalyst ID to determine  """
        self.predictions = cross_val_predict(self.machina, self.features_df.values, self.labels_df.values,
                                             groups=self.groups, cv=GroupKFold(kfold))

        if add_to_slave:
            self.slave_dataset['predictions'] = self.predictions
            self.slave_dataset.to_csv('{}//{}-slave.csv'.format(self.svfl, self.svnm))

    def predict_leaveoneout(self, add_to_slave=False):
        self.predictions = cross_val_predict(self.machina, self.features_df.values, self.labels_df.values,
                                             groups=self.groups, cv=LeaveOneGroupOut())

        if add_to_slave:
            self.slave_dataset['predictions'] = self.predictions
            self.slave_dataset.to_csv('{}//{}-slave.csv'.format(self.svfl, self.svnm))

    def predict_leave_yoself_out(self, add_to_slave=False):
        self.predictions = cross_val_predict(self.machina, self.features_df.values, self.labels_df.values,
                                             cv=LeaveOneOut())

        if add_to_slave:
            self.slave_dataset['predictions'] = self.predictions
            self.slave_dataset.to_csv('{}//{}-slave.csv'.format(self.svfl, self.svnm))

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

    # TODO Depricated - Remove when predict_from_catalyst_ids is working
    def create_test_dataset(self, catids):
        """
        Create a test dataset from slave, drop catalysts from slave
        This allows for the ML algorithm to be trained on slave and predict the test dataset blindly
        """
        self.tester_dataset = self.slave_dataset[self.slave_dataset.index.isin(catids)].copy()
        self.slave_dataset.drop(labels=self.tester_dataset.index, inplace=True)
        self.set_training_data() # This rewrites features and labels dataframes with slave

    def predict_within_elements(self, elements, svnm='data'):
        element_dataframe = pd.DataFrame()

        for ele in elements:
            dat = self.slave_dataset.loc[self.slave_dataset['{} Loading'.format(ele)] > 0]
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
        element_dataframe = pd.DataFrame()

        for jj, ele in enumerate(elements):
            if loads is None:
                dat = self.slave_dataset.loc[self.slave_dataset['{} Loading'.format(ele)] > 0]
            else:
                dat = self.slave_dataset.loc[self.slave_dataset['{} Loading'.format(ele)] == loads[jj]]

            element_dataframe = pd.concat([element_dataframe, dat])

        drop_ids = np.unique(element_dataframe.index)

        self.labels_df = element_dataframe.loc[:, 'Measured Conversion'].copy()

        self.features_df = element_dataframe.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'group'],
                axis=1
            )

        self.train_data()

        test_data = self.slave_dataset.drop(index=drop_ids).copy()

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

    def predict_from_catalyst_ids(self):
        pass

    # TODO Depricated - Remove when predict_from_catalyst_ids is working
    def predict_from_masterfile(self, catids, svnm='data', temp_slice=True):
        """ Description """
        # Note: This may break due to significant changes in the learner methods

        self.create_test_dataset(catids)
        self.train_data()

        """ Comment - Work in Progress """
        data = self.tester_dataset.drop(
            labels=['Measured Conversion', 'Element Dictionary', 'group'],
            axis=1
        ).values

        print(self.tester_dataset)
        print(data)

        predvals = self.machina.predict(data)

        original_test_df = self.slave_dataset.loc[self.tester_dataset.index].copy()
        measvals = original_test_df.loc[:, 'Measured Conversion'].values

        comparison_df = pd.DataFrame([predvals, measvals],
                           index=['Predicted Conversion','Measured Conversion'],
                           columns=original_test_df.index).T

        comparison_df['ID'] = comparison_df.index   #[x.split('_')[0] for x in comparison_df.index]
        comparison_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in self.tester_dataset['Element Dictionary']
        ]
        comparison_df['temperature'] = original_test_df['temperature']

        # I'm not entirely sure why I'm dropping 400 and 450, unless it's because I arbitrarily want to see 350orless
        comparison_df.drop(comparison_df[comparison_df.loc[:, 'temperature'] == 450].index, inplace=True)
        comparison_df.drop(comparison_df[comparison_df.loc[:, 'temperature'] == 400].index, inplace=True)

        feat_df = self.extract_important_features()
        feat_df.to_csv('{}\\{}-features.csv'.format(self.svfl, svnm))

        g = Graphic(learner=self, df=comparison_df)
        g.plot_err(svnm='{}-predict_{}'.format(self.version, svnm))
        g.plot_err(svnm='{}-predict_{}_nometa'.format(self.version, svnm), metadata=False)
        g.plot_important_features(svnm=svnm)
        g.bokeh_predictions(svnm='{}-predict_{}'.format(self.version, svnm))

    def predict_from_master_dataset(self):
        """ Description """
        # Note: This may break due to significant changes in the learner methods

        data = self.master_dataset[self.master_dataset.index.str.contains('Predict') == True]
        data = data.drop(
            labels=['Measured Conversion', 'Element Dictionary', 'group'], axis=1
        )

        predvals = self.machina.predict(data.values)
        data['Predictions'] = predvals
        data.to_csv(r'{}/{}-BinaryPredictions.csv'.format(self.svfl, self.version))
        return data

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
            df = pd.DataFrame(
                np.array([
                    self.plot_df.index,
                    self.predictions,
                    self.labels_df.values,
                    self.groups,
                    self.plot_df['Name'],
                    self.plot_df['temperature']]).T,
                columns=['ID', 'Predicted Conversion', 'Measured Conversion', 'Groups', 'Name', 'Temperature'])
            df.to_csv('{}\predictions-{}.csv'.format(self.svfl, self.svnm))
        else:
            print('No predictions to save...')

    def save_slave(self):
        """ Comment """
        self.slave_dataset.to_csv('{}\slavedata-{}.csv'.format(self.svfl, self.svnm))

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

