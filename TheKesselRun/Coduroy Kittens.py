# Created by Travis Williams
# Property of the University of South Carolina
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, BoundaryNorm, to_hex, Normalize
from matplotlib.cm import get_cmap

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict, GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, \
    roc_curve, recall_score, precision_score, mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.kernel_ridge import KernelRidge

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker, CustomJS, Slider, Select
from bokeh.plotting import figure, show, output_file, save, curdoc
import bokeh.palettes as pals
from bokeh.models import Range1d, DataRange1d
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.palettes import Plasma
from bokeh.sampledata.autompg import autompg_clean

import re
import itertools

import seaborn as sns
import ast
import graphviz
import os
from itertools import compress
import time


class Learner():
    """SupervisedLearner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self, average_data=True, element_filter=3, temperature_filter=None, group_style='blind', version='v00',
                 feature_generator=0, regression=True):
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

        '''Initialize ML algorithm and tuning parameters'''
        self.machina = None

        '''Initialize all options for the algorithm.  These are used in naming files.'''
        self.average_data = average_data
        self.element_filter = element_filter
        self.temperature_filter = temperature_filter
        self.group_style = group_style
        self.version = version
        self.feature_generator = feature_generator
        self.regression = regression

        '''Create path, folder, and filename'''
        self.svfl = './/Results//{version}_{type}'.format(version=version, type='r' if regression else 'c')
        self.svnm = '{nm}_{type}-{nele}-{temp}-{grp}-{feat}'.format(
            nm=version,
            type = 'r' if regression else 'c',
            nele=element_filter,
            temp='{}C'.format(temperature_filter) if temperature_filter is not None else 'All',
            grp=group_style,
            feat=feature_generator
        )

        if not os.path.exists(self.svfl):
            os.makedirs(self.svfl)
            os.makedirs('{}\\{}'.format(self.svfl, 'trees'))
            os.makedirs('{}\\{}'.format(self.svfl, 'figures'))
            os.makedirs('{}\\{}'.format(self.svfl, 'htmls'))

        '''Initialize Time for run-length statistics'''
        self.start_time = time.time()

    def set_name_paths(self):
        self.svfl = './/Results//{version}_{type}'.format(version=self.version, type='r' if self.regression else 'c')
        self.svnm = '{nm}_{type}-{nele}-{temp}-{grp}-{feat}'.format(
            nm=self.version,
            type='r' if self.regression else 'c',
            nele=self.element_filter,
            temp='{}C'.format(self.temperature_filter) if self.temperature_filter is not None else 'All',
            grp=self.group_style,
            feat=self.feature_generator
        )

    def set_temp_filter(self, temp_filter):
        """ Set Temperature Filter to allow multiple slices without reloading the data """
        self.temperature_filter = temp_filter
        self.set_name_paths()

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

    def load_nh3_catalysts(self):
        """ Import NH3 data from Katie's HiTp dataset(cleaned). """

        if self.average_data:
            df = pd.read_csv(r".\Data\Processed\AllData_Condensed.csv", index_col=0)
        else:
            df = pd.read_csv(r".\Data\Processed\AllData.csv", index_col=0)

        for index, row in df.iterrows():
            cat = Catalyst()
            cat.ID = row['ID']
            cat.add_element(row['Ele1'], row['Wt1'])
            cat.add_element(row['Ele2'], row['Wt2'])
            cat.add_element(row['Ele3'], row['Wt3'])
            cat.input_reactor_number(int(row['Reactor']))
            cat.input_temperature(row['Temperature'])
            cat.input_space_velocity(row['Space Velocity'])
            cat.input_ammonia_concentration(row['NH3'])
            if self.average_data:
                cat.input_standard_error(row['Standard Error'])
                cat.input_n_averaged_samples(row['nAveraged'])
            cat.activity = row['Concentration']
            cat.feature_add_n_elements()
            # cat.feature_add_oxidation_states()

            feature_generator = {
                0: cat.feature_add_elemental_properties,
                1: cat.feature_add_statistics,
                2: cat.feature_add_weighted_average
            }
            feature_generator.get(self.feature_generator, lambda: print('No Feature Generator Selected'))()

            self.add_catalyst(index='{ID}_{T}'.format(ID=cat.ID, T=row['Temperature']), catalyst=cat)

        self.create_master_dataset()

    def create_master_dataset(self):
        # Set up catalyst loading dictionary with loadings
        loading_df = pd.read_csv('.\\Data\\Elements.csv', usecols=['Abbreviation'], index_col='Abbreviation').transpose()
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

        # 1. Filter Based on Number of Elements
        filter_dict_neles = {
            1: self.master_dataset[self.master_dataset['n_elements'] == 1].index,
            2: self.master_dataset[self.master_dataset['n_elements'] == 2].index,
            3: self.master_dataset[self.master_dataset['n_elements'] == 3].index
        }

        if self.element_filter is list():
            n_ele_slice = filter_dict_neles.get(self.element_filter, self.master_dataset.index)
        else:
            n_ele_slice = filter_dict_neles.get(self.element_filter, self.master_dataset.index)

        # 2. Filter based on temperature
        if self.temperature_filter is None:
            temp_slice = self.master_dataset[self.master_dataset.loc[:, 'temperature'] != 150].index
        elif isinstance(self.temperature_filter, str):
            temp_dict = {
                'not450': self.master_dataset[(self.master_dataset.loc[:, 'temperature'] != 450) &
                                              (self.master_dataset.loc[:, 'temperature'] != 150)].index
            }

            temp_slice = temp_dict.get(self.temperature_filter)
        else:
            temp_slice = self.master_dataset[self.master_dataset.loc[:, 'temperature'] == self.temperature_filter].index

        # 3. Create the filter (filter is used to slice the master file)
        def join_all_indecies(ind_list):
            start_filter = ind_list.pop(0)
            for ind_obj in ind_list:
                start_filter = list(set(start_filter) & set(ind_obj))
            return start_filter

        filt = join_all_indecies([n_ele_slice, temp_slice])

        # 4. Need to write a method to remove other features
        def drop_element(ele, filt):
            slice = self.master_dataset[self.master_dataset['{} Loading'.format(ele)] == 0].index
            return join_all_indecies([filt, slice])

        # 5. Drop Tungston Data because it's always bad
        # filt = drop_element('W', filt)

        # 6. Apply filter, master to slave, shuffle slave
        self.slave_dataset = self.master_dataset.loc[filt].copy()
        self.slave_dataset = self.slave_dataset[self.slave_dataset.index.str.contains('Predict') == False]
        self.slave_dataset = shuffle(self.slave_dataset)
        # pd.DataFrame(self.slave_dataset).to_csv('.\\SlaveTest.csv')

        # 7. Set up training data and apply grouping
        self.set_training_data()
        self.group_for_training()

    def set_training_data(self):
        ''' Use the slave dataframe to set other dataframe properties '''

        if self.average_data:
            self.features_df = self.slave_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged'], axis=1
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
            'blind': [x.split('_')[0] for x in self.slave_dataset.index.values],
            'semiblind': ['{}_{}'.format(x.split('_')[0], x.split('_')[1]) for x in self.slave_dataset.index.values]
        }

        self.groups = group_dict.get(self.group_style, None)

    def hyperparameter_tuning(self):
        """ Comment """
        # gs = GridSearchCV(self.machina, self.machina_tuning_parameters, cv=10, return_train_score=True)
        gs = RandomizedSearchCV(self.machina, self.machina_tuning_parameters, cv=GroupKFold(10),
                                return_train_score=True, n_iter=500)
        gs.fit(self.features_df.values, self.labels_df.values, groups=self.groups)
        pd.DataFrame(gs.cv_results_).to_csv('{}\\p-tune_{}.csv'.format(self.svfl, self.svnm))

    def set_learner(self, learner, params='default'):
        """ Comment """
        learn_selector = {
            'rfr': RandomForestRegressor,
            'rfc': RandomForestClassifier,
            'adaboost': AdaBoostRegressor,
            'tree': tree.DecisionTreeRegressor,
            'SGD': None,
            'neuralnet': MLPRegressor,
            'svr': SVR,
            'knnr': KNeighborsRegressor,
            'krr': KernelRidge,
            'etr': ExtraTreesRegressor,
            'gbr': GradientBoostingRegressor
        }

        if self.regression:
            param_selector = {
                'default': {'n_estimators':100, 'max_depth':None, 'min_samples_leaf':2, 'min_samples_split':2,
                            'max_features':'auto', 'bootstrap':True, 'n_jobs':4, 'criterion':'mse'},
                'v1': {'n_estimators':50, 'max_depth':None, 'min_samples_leaf':2, 'min_samples_split':2},
                'v2': {'n_estimators': 50, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2,
                            'max_features': 'auto', 'bootstrap': True},
                'v3': {'n_estimators': 250, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                            'max_features': 'sqrt', 'bootstrap': False},
                'adaboost': {'base_estimator':RandomForestRegressor(), 'n_estimators':1000},
                'nnet': {'hidden_layer_sizes':100, 'solver':'sgd'},
                'knnr': {'n_neighbors': 3, 'weights': 'distance'},
                'svr': {},
                'empty': {},
                'test': {'n_estimators':250, 'max_depth':None, 'min_samples_leaf':2, 'min_samples_split':2,
                            'max_features':'auto', 'bootstrap':True, 'n_jobs':4, 'criterion':'mae'},
            }
        else:
            param_selector = {
                'default': {'n_estimators':100, 'n_jobs':4}
            }

        self.machina = learn_selector.get(learner, lambda: 'Error')()
        self.machina.set_params(**param_selector.get(params))

    def train_data(self):
        """ Comment """
        self.machina = self.machina.fit(self.features_df.values, self.labels_df.values)

    def predict_dataset(self):
        data = self.master_dataset[self.master_dataset.index.str.contains('Predict')].copy()
        data = data.drop(
            labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged'], axis=1
        )
        predvals = self.machina.predict(data.values)
        data['Predictions'] = predvals
        data.to_csv(r'./BinaryPredictions-v13.csv')

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

    def predict_from_masterfile(self, catids, svnm='data'):
        """ Descr """
        self.create_test_dataset(catids)
        self.train_data()

        """ Comment - Work in Progress """
        if self.average_data:
            data = self.test_dataset.drop(
                labels=['Measured Conversion', 'Element Dictionary', 'standard error', 'n_averaged'],
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

        rats = np.abs(np.subtract(predvals, measvals, out=np.zeros_like(predvals),
                                  where=measvals != 0))

        rat_count = rats.size
        wi5 = (rats < 0.05).sum()
        wi10 = (rats < 0.10).sum()
        wi20 = (rats < 0.20).sum()

        x = np.array([0, 0.5, 1])
        y = np.array([0, 0.5, 1])

        fig, ax = plt.subplots()
        ax.plot(x, y, lw=2, c='k')
        ax.fill_between(x, y + 0.1, y - 0.1, alpha=0.1, color='b')
        ax.fill_between(x, y + 0.2, y + 0.1, alpha=0.1, color='y')
        ax.fill_between(x, y - 0.2, y - 0.1, alpha=0.1, color='y')
        ax.text(0.75, 0.05, s='Within 5%: {five:0.2f} \nWithin 10%: {ten:0.2f} \nWithin 20%: {twenty:0.2f}'.format(
            five=wi5 / rat_count, ten=wi10 / rat_count, twenty=wi20 / rat_count))

        comparison_df.to_csv('.\\Results\\Predictions\\predict_{}.csv'.format(svnm))
        comparison_df.plot(x='Predicted Conversion', y='Measured Conversion', kind='scatter', ax=ax)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig('.\\Results\\Predictions\\predict_{}.png'.format(svnm), dpi=400)
        plt.close()

    def predict_crossvalidate(self):
        """ Comment """
        self.predictions = cross_val_predict(self.machina, self.features_df.values, self.labels_df.values,
                                             groups=self.groups, cv=GroupKFold(10))

    def save_predictions(self):
        """ Comment """
        if self.predictions is not None:
            df = pd.DataFrame(np.array([self.slave_dataset.index, self.predictions, self.labels_df.values]).T,
                              columns=['ID', 'Predicted Conversion', 'Measured Conversion'])
            df.to_csv('{}\predictions-{}.csv'.format(self.svfl, self.svnm))
        else:
            print('No predictions to save...')

    def extract_important_features(self, sv=False, prnt=False):
        """ Save all feature importance, print top 10 """

        try:
            df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns,
                          columns=['Feature Importance'])
        except AttributeError:
            return

        if prnt:
            print(df.sort_values(by='Feature Importance', ascending=False).head(10))

        if sv:
            df.to_csv('{}//feature_importance-{}.csv'.format(self.svfl, self.svnm))
        else:
            return df

    def evaluate_regression_learner(self):
        """ Comment """
        r2 = r2_score(self.labels_df.values, self.predictions)
        mean_abs_err = mean_absolute_error(self.labels_df.values, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.labels_df.values, self.predictions))

        print('\n----- Model {} -----'.format(self.svnm))
        print('R2: {:0.3f}'.format(r2))
        print('Mean Absolute Error: {:0.3f}'.format(mean_abs_err))
        print('Mean Squared Error: {:0.3f}'.format(rmse))
        print('Time to Complete: {:0.1f} s'.format(time.time() - self.start_time))
        print('\n')

        pd.DataFrame([r2, mean_abs_err, rmse, time.time() - self.start_time],
                     index=['R2','Mean Abs Error','RMSE','Time']).to_csv('{}\\{}-eval.csv'.format(self.svfl, self.svnm))

    def evaluate_classification_learner(self):
        print(self.predictions)
        fpr, tpr, thershold = roc_curve(self.labels_df.values, self.predictions)
        print(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.show()

    def preplotcessing(self):
        """ Prepare all data for plotting """

        # Ensure Predictions Exist
        if self.predictions is None:
            self.predict_crossvalidate()

        # Set up the plot dataframe for easy plotting (specifically for Bokeh)
        self.plot_df = self.slave_dataset.copy()
        self.plot_df['Predicted Conversion'] = self.predictions

        # Full descriptive name X(#)Y(#)Z(#)
        self.plot_df['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in self.plot_df['Element Dictionary']
        ]

        # Second Element Name
        self.plot_df['Ele2'] = [
            ''.join('{}'.format(key) if (key != 'Ru') & (key != 'K') else ''
                    for key, val in x) for x in self.plot_df['Element Dictionary']
        ]

        # Second Element Weight Loading
        self.plot_df['Load2'] = [
            ''.join('{}'.format(val) if (key != 'Ru') & (key != 'K') else ''
                    for key, val in x) for x in self.plot_df['Element Dictionary']
        ]

        # CatalystObject ID
        self.plot_df['ID'] = [int(nm.split('_')[0]) for nm in self.plot_df.index.values]

        # Remove Dictionary to avoid problems down the line
        self.plot_df.drop(columns='Element Dictionary', inplace=True)

        # Create hues for heatmaps
        def create_feature_hues(self, feature):
            try:
                unique_feature = np.unique(self.slave_dataset.loc[:, feature].values)
            except KeyError:
                print('KeyError: {} not found'.format(feature))
                return

            n_feature = len(unique_feature)
            max_feature = np.max(unique_feature)
            min_feature = np.min(unique_feature)

            if max_feature == min_feature:
                self.plot_df['{}_hues'.format(feature)] = "#3498db"  # Blue!
            else:
                palette = sns.color_palette('plasma', n_colors=n_feature+1)
                self.plot_df['{}_hues'.format(feature)] = [
                    palette[i] for i in [int(n_feature * (float(x) - min_feature) / (max_feature - min_feature))
                                              for x in self.slave_dataset.loc[:, feature].values]
                ]

        self.plot_df['temperature_hues'] = 0

        # Grab top 10 features, add hues to plotdf
        try:
            feature_rank = self.extract_important_features()
            for feat in feature_rank.sort_values(by='Feature Importance', ascending=False).head(10).index.values:
                create_feature_hues(self, feat)
        except AttributeError:
            print('SupervisedLearner does not support feature extraction.')

        # Process Second Element Colors
        uniq_eles = np.unique(self.plot_df['Ele2'])
        n_uniq = len(uniq_eles)
        palette = sns.color_palette('tab20', n_colors=n_uniq + 1)
        self.plot_df['Ele2_hues'] = [
            palette[np.where(uniq_eles == i)[0][0]] for i in self.plot_df['Ele2']
        ]

        return self.plot_df

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

    def plot_error(self):
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
        ax.text(0.75, 0.05, s='Within 5%: {five:0.2f} \nWithin 10%: {ten:0.2f} \nWithin 20%: {twenty:0.2f}'.format(
            five=wi5 / rat_count, ten=wi10 / rat_count, twenty=wi20 / rat_count))

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


class Catalyst():
    """CatalystObject will contain each individual training set"""
    def __init__(self):
        self.ID = None
        self.activity = None

        self.input_dict = dict()
        self.feature_dict = dict()
        self.elements = dict()

    def input_temperature(self, T):
        self.input_dict['temperature'] = T

    def add_element(self, element, weight_loading):
        if (element != '-') & (element != '--'):
            self.elements[element] = weight_loading

    def input_space_velocity(self, space_velocity):
        self.input_dict['space_velocity'] = space_velocity

    def input_reactor_number(self, reactor_number):
        self.input_dict['reactor_number'] = reactor_number

    def input_ammonia_concentration(self, ammonia_concentration):
        self.input_dict['ammonia_concentration'] = ammonia_concentration

    def input_standard_error(self, error):
        self.input_dict['standard error'] = error

    def input_n_averaged_samples(self, n_avg):
        self.input_dict['n_averaged'] = n_avg

    def feature_add(self, key, value):
        self.feature_dict[key] = value

    def feature_add_statistics(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements.keys())]

        for prop in eledf:
            self.feature_add('{}_mean'.format(prop), eledf.loc[:, prop].mean())
            self.feature_add('{}_mad'.format(prop), eledf.loc[:, prop].mad())

    def feature_add_weighted_average(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements.keys())]

        def calc_weighted_average(a, b):
            num = np.sum(a * b)
            den = np.sum(b)
            return num/den

        for feature_name, feature_values in eledf.T.iterrows():
            feat = calc_weighted_average(feature_values.values, np.fromiter(self.elements.values(), dtype=float))
            self.feature_add('{nm}_wtavg'.format(nm=feature_name), feat)

    def feature_add_elemental_properties(self):
        # Load Elements.csv as DataFrame, Slice Elements.csv based on elements present
        eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)
        eledf = eledf.loc[list(self.elements.keys())]

        # Methods of processing different features
        def calc_weighted_average(self, values, weights, feature_name):
            numor = np.sum(values * weights)
            denom = np.sum(weights)
            self.feature_add('{nm}_wtavg'.format(nm=feature_name), numor/denom)

        def calc_statistics(self, values, weights, feature_name):
            self.feature_add('{}_mean'.format(feature_name), np.mean(values))
            self.feature_add('{}_mad'.format(feature_name), np.mean(np.abs(values-np.mean(values))))
            # self.feature_add('{}_med'.format(feature_name), np.median(values))
            # self.feature_add('{}_min'.format(feature_name), np.max(values))
            # self.feature_add('{}_max'.format(feature_name), np.min(values))
            # self.feature_add('{}_rng'.format(feature_name), np.max(values)-np.min(values))

        # Create Dictionary to process each feature differently
        process_dict = {
            'Atomic Number': calc_statistics,
            # 'Abbreviation': None,
            'Atomic Volume': calc_weighted_average,
            'Atomic Weight': calc_weighted_average,
            'Boiling Temperature': calc_weighted_average,
            'Periodic Table Column': calc_statistics,
            'Covalent Radius': calc_weighted_average,
            'Density': calc_weighted_average,
            'Dipole Polarizability': calc_weighted_average,
            'Electron Affinity': calc_weighted_average,
            'Electronegativity': calc_statistics,
            'Fusion Enthalpy': calc_weighted_average,
            'GS Bandgap': calc_statistics,
            'GS Energy': calc_statistics,
            'Heat Capacity (Mass)': calc_weighted_average,
            'Heat Capacity (Molar)': calc_weighted_average,
            'Heat Fusion': calc_weighted_average,
            'First Ionization Energy': calc_statistics,
            'Second Ionization Energy': calc_statistics,
            'Third Ionization Energy': calc_statistics,
            'Fourth Ionization Energy': calc_statistics,
            'Fifth Ionization Energy': calc_statistics,
            'Sixth Ionization Energy': calc_statistics,
            'Seventh Ionization Energy': calc_statistics,
            'Eighth Ionization Energy': calc_statistics,
            'IsAlkali': calc_statistics,
            'IsDBlock': calc_statistics,
            'IsFBlock': calc_statistics,
            'IsMetal': calc_statistics,
            'IsMetalloid': calc_statistics,
            'IsNonmetal': calc_statistics,
            'Melting Temperature': calc_weighted_average,
            'Mendeleev Number': calc_statistics,
            'Number d-shell Unfilled Electrons': calc_statistics,
            'Number d-shell Valance Electrons': calc_statistics,
            'Number f-shell Unfilled Electrons': calc_statistics,
            'Number f-shell Valance Electrons': calc_statistics,
            'Number p-shell Unfilled Electrons': calc_statistics,
            'Number p-shell Valance Electrons': calc_statistics,
            'Number s-shell Unfilled Electrons': calc_statistics,
            'Number s-shell Valance Electrons': calc_statistics,
            'Number Unfilled Electrons': calc_statistics,
            'Number Valence Electrons': calc_statistics,
            'Polarizability': calc_statistics,
            'Periodic Table Row': calc_statistics,
            'Zunger Pseudopotential (d)': calc_statistics,
            'Zunger Pseudopotential (p)': calc_statistics,
            'Zunger Pseudopotential (pi)': calc_statistics,
            'Zunger Pseudopotential (s)': calc_statistics,
            'Zunger Pseudopotential (sigma)': calc_statistics,
            'phi': calc_statistics,
            'Conductivity': calc_statistics
        }

        for feature_name, feature_values in eledf.T.iterrows():
            process_dict.get(feature_name,
                             lambda a,b,c,d: print('Feature Name ({}) Not Found'.format(feature_name)))(
                self,
                feature_values,
                np.fromiter(self.elements.values(), dtype=float),
                feature_name
            )

    def feature_add_n_elements(self):
        n_eles = 0
        for val in self.elements.values():
            if val > 0:
                n_eles += 1

        self.feature_add('n_elements',n_eles)

    def feature_add_M1M2_ratio(self):
        if len(list(self.elements.values())) >= 2:
            ratio = list(self.elements.values())[0] / list(self.elements.values())[1] * 100
        else:
            ratio = 0
        self.feature_add('M1M2_ratio', ratio)

    def feature_add_oxidation_states(self):
        eledf = pd.read_csv(r'./Data/Elements.csv', index_col=0, usecols=['Abbreviation','OxidationStates'])
        eledf.dropna(inplace=True)
        eledf = eledf.loc[list(self.elements.keys())]

        for indx, val in eledf.iterrows():
            for ox_state in val.values[0].split(' '):
                eledf.loc[indx, 'OxState {}'.format(ox_state)] = 1

        eledf.fillna(0, inplace=True)
        eledf.drop(columns='OxidationStates', inplace=True)

        for feature_name, feature_values in eledf.T.iterrows():
            for index, _ in enumerate(self.elements):
                self.feature_add('{nm}_{index}'.format(nm=feature_name, index=index),
                                 feature_values.values[index])


def standard_pipeline(learner):
    learner.filter_static_dataset()
    learner.train_data()
    learner.extract_important_features(sv=True, prnt=True)
    learner.predict_crossvalidate()
    if learner.regression:
        learner.evaluate_regression_learner()
    else:
        learner.evaluate_classification_learner()
    learner.compile_results()
    learner.plot_basic()
    learner.plot_error()


def temperature_slice(learner):
    for t in [250, 300, 350, 400, 450, None, 'not450']:
        learner.set_temp_filter(t)
        learner.filter_static_dataset()
        learner.train_data()
        learner.extract_important_features(sv=True, prnt=True)
        learner.predict_crossvalidate()
        if learner.regression:
            learner.evaluate_regression_learner()
        else:
            learner.evaluate_classification_learner()
        learner.save_predictions()
        learner.compile_results()
        learner.plot_basic()
        learner.plot_error()
        learner.plot_features_colorbar(x_feature='Predicted Conversion', c_feature='ammonia_concentration')
        learner.bokeh_predictions()
        learner.bokeh_by_elements()

def predict_all_binaries():
    def create_catalyst(e1, w1, e2, w2, e3, w3, tmp, reactnum, space_vel, ammonia_conc):
        cat = Catalyst()
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
        temperature_filter=None,
        group_style='blind',  # blind groups by catalyst ID, semiblind groups by temperatures within catalyst ID
        version='v13-pred',
        feature_generator=0,  # 0 is elemental, 1 is statistics,  2 is statmech
        regression=True
    )
    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)
    ele_list = [12] + list(range(20, 31)) + list(range(38, 43)) + list(range(44, 51)) + list(range(74, 80)) + [56,72,82,83]
    ele_df = eledf[eledf['Atomic Number'].isin(ele_list)]
    eles = ele_df.index.values
    combos = list(itertools.combinations(eles, r=2))

    final_list = list()
    for vals in combos:
        tmp=300

        cat1 = create_catalyst(e1=vals[0], w1=3, e2=vals[1], w2=1, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat1)

        cat2 = create_catalyst(e1=vals[0], w1=2, e2=vals[1], w2=2, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat2)

        cat3 = create_catalyst(e1=vals[0], w1=1, e2=vals[1], w2=3, e3='K', w3=12,
                               tmp=tmp, reactnum=8, space_vel=2000, ammonia_conc=0.01)
        skynet.add_catalyst('Predict', cat3)

    skynet.load_nh3_catalysts()
    skynet.filter_master_dataset()
    skynet.train_data()
    skynet.predict_dataset()


if __name__ == '__main__':
    # predict_all_binaries()
    # exit()

    # Begin Machine Learning
    skynet = Learner(
        average_data=True,
        element_filter=3,
        temperature_filter=None,
        group_style='blind', # blind groups by catalyst ID, semiblind groups by temperatures within catalyst ID
        version='v16-rfr',
        feature_generator=0, # 0 is elemental, 1 is statistics,  2 is statmech
        regression=True
    )
    if skynet.regression:
        skynet.set_learner(learner='rfr', params='default')
    else:
        skynet.set_learner(learner='rfc', params='default')

    skynet.load_nh3_catalysts()
    # temperature_slice(learner=skynet)
    # # # standard_pipeline(learner=skynet)
    # exit()

    skynet.filter_master_dataset()
    skynet.predict_from_masterfile(catids=[65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 82, 83], svnm='SS8')
    skynet.predict_from_masterfile(catids=[38, 84, 85, 86, 87, 89, 90, 91, 93], svnm='SS9')
    exit()
    skynet.train_data()
    skynet.extract_important_features(sv=True)
    skynet.predict_crossvalidate()
    skynet.evaluate_regression_learner()
    pltdf = skynet.preplotcessing()

    # skynet.bokeh_test()
    # skynet.plot_visualize_error()
    # exit()
    # skynet.visualize_tree(n=1)
    skynet.plot_basic()
    skynet.plot_error()
    # skynet.plot_features(x_feature='Ru Loading', c_feature='temperature')


    # skynet.bokeh_important_features(svtag='IonEn_{}'.format(skynet.temp_filter),
    #                                 xaxis='Measured Conversion', xlabel='Measured Conversion', xrng=None,
    #                                 yaxis='Predicted Conversion', ylabel='Predicted Conversion', yrng=None,
    #                                 caxis='IonizationEnergies_2_1')
    # skynet.bokeh_predictions()

    # for tp in [250, 300, 350]:
    #     # cols = ['NdValence_1', "IonizationEnergies_2_1", 'Column_1']
    #     # col_nms = ['Number of d-Valence Electrons', 'Second Ionization Energy', 'Column']
    #
    #     cols = ['FusionEnthalpy_mean', 'IonizationEnergies_2_mad', 'HeatFusion_mean', 'ZungerPP-r_d_mean']
    #     col_nms = ['Mean Fusion Enthalpy','MAD 2nd Ionization Energy','Mean Heat of Fusion','Mean Zunger d Radius']
    #
    #     for index, ftr in enumerate(cols):
    #         nm = col_nms[index]
    #         skynet.plot_important_features_bokeh(
    #             temp_slice=tp, svtag='{}-{}'.format(ftr, tp),
    #             xaxis=ftr, xlabel=nm, xrng=DataRange1d(),
    #             yaxis='Measured', ylabel='Measured Conversion', yrng=Range1d(0,1)
    #         )
    # skynet.save_predictions()