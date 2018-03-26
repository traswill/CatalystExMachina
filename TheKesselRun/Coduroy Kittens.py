# Created by Travis Williams
# Property of the University of South Carolina
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict, GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker
from bokeh.plotting import figure, show, output_file, save
import bokeh.palettes as pals
from bokeh.models import Range1d, DataRange1d

import seaborn as sns
import ast
import graphviz
import os
from itertools import compress
import time


class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self, import_file='AllData', svfl='.\\Results', svnm='Test', feature_type=0):
        self.catalyst_dictionary = dict()

        self.master_dataset = pd.DataFrame()
        self.slave_dataset = pd.DataFrame()
        self.test_dataset = pd.DataFrame()

        self.features_df = pd.DataFrame()
        self.plot_df = pd.DataFrame()
        self.features = list()
        self.labels = list()
        self.predictions = list()

        self.machina = None
        self.machina_tuning_parameters = None

        self.feature_type = feature_type
        self.filter = None
        self.groups = None

        self.svfl = svfl
        self.svnm = svnm
        if not os.path.exists(self.svfl):
            os.makedirs(self.svfl)
            os.makedirs('{}\\{}'.format(self.svfl, 'trees'))
            os.makedirs('{}\\{}'.format(self.svfl, 'figures'))
            os.makedirs('{}\\{}'.format(self.svfl, 'htmls'))

        self.impfl = import_file
        self.start_time = time.time()

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
        df = pd.read_csv(r".\Data\Processed\{}.csv".format(self.impfl), index_col=0)

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
            cat.activity = row['Concentration']

            cat.feature_add_n_elements()

            feature_generator = {
                0: cat.feature_add_elemental_properties,
                1: cat.feature_add_statistics
            }
            feature_generator.get(self.feature_type, lambda: print('No Feature Generator Selected'))()


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
            actdf = pd.DataFrame(catobj.activity, index=[catid], columns=['Measured Activity'])

            # Create element dictionary
            eldictdf = pd.DataFrame(catobj.elements.items(), index=[catid], columns=['Element Dictionary'])

            # Combine DFs
            df = pd.concat([load_df, inputdf, featdf, actdf, eldictdf], axis=1)
            self.master_dataset = pd.concat([self.master_dataset, df], axis=0)

        self.master_dataset.dropna(how='all', axis=1, inplace=True)
        self.master_dataset.fillna(value=0, inplace=True)

        print(self.master_dataset)

    def filter_master_dataset(self, filter=None, temperature=None, group=None, features=None):
        """ Filters Data from import file for partitioned model training

        :: filter :: (string)
            mono - only predict with monometallics
            bi - only pridict with bimetallics
            3ele - only predict with Ru-X-K catalysts
        :: temperature :: (int)
            Partition to only predict with catalysts at designated temperature
        :: group :: (string)
            Determines how data is partitioned into 10-fold validation groups.
            blind - Catalyst ID cannot be in both test and training set
            semiblind - Catalyst ID_Temp cannot be in both test and training set
        :: features ::
            Not Yet Implemented
        """

        def filter_elements(filter):
            if filter == 'mono':
                dat_filter = self.master_dataset[self.master_dataset['n_elements'] != 1].index

            elif filter == 'bi':
                dat_filter = self.master_dataset[self.master_dataset['n_elements'] != 2].index

            elif filter == '3ele':
                dat_filter = self.master_dataset[self.master_dataset['n_elements'] != 3].index
            else:
                dat_filter = []

            return dat_filter

        def filter_temp(temperature):
            if temperature is not None:
                temp_filter = self.master_dataset[self.master_dataset.loc[:, 'temperature'] != temperature].index
            else:
                temp_filter = []

            return temp_filter

        def filter_features(feats):
            # Not implemented
            return []

        def filter_bad_data():
            bad_filter = self.master_dataset[self.master_dataset['W Loading'] > 0].index
            return bad_filter

        self.filter = np.union1d(filter_elements(filter),
                                 filter_temp(temperature)
                                 )

        self.filter = np.union1d(self.filter, filter_bad_data())

        # Filter Master to Slave
        self.slave_dataset = self.master_dataset.drop(index=self.filter).copy()
        self.slave_dataset = shuffle(self.slave_dataset)
        # pd.DataFrame(self.slave_dataset).to_csv('.\\SlaveTest.csv')

        self.set_training_data()

    def set_training_data(self):
        # Set all other DFs from slave
        self.features_df = self.slave_dataset.drop(labels=['Measured Activity', 'Element Dictionary'], axis=1).copy()
        self.labels_df = self.slave_dataset['Measured Activity'].copy()
        self.plot_df = self.slave_dataset.loc[:, ['Measured Activity', 'Element Dictionary']].copy()

        # Set Features and Labels
        self.features = self.features_df.values
        self.labels = self.labels_df.values

        self.group_for_training(group=group)

    def create_test_dataset(self, catids):
        # Create Temporary indexer to slice slave dataset
        ind = [int(idtag.split('_')[0]) for idtag in self.slave_dataset.index]
        self.slave_dataset['temp_ind'] = ind

        # Slice the dataset, copying all values of catids
        self.test_dataset = self.slave_dataset[self.slave_dataset['temp_ind'].isin(catids)].copy()

        # Drop the temporary indexer
        self.slave_dataset.drop(columns=['temp_ind'], inplace=True)
        self.test_dataset.drop(columns=['temp_ind'], inplace=True)

        # Remove test dataset from slave dataset to prepare for training
        self.slave_dataset.drop(labels=self.test_dataset.index, inplace=True)

        self.set_training_data()

    def group_for_training(self, group=None):
        """ Comment """
        if group == 'blind':
            # Group by Sample ID
            self.groups = [x.split('_')[0] for x in self.slave_dataset.index.values]
        elif group == 'semiblind':
            # Group by Sample ID and Temperature (allow same element-different T in training sets)
            self.groups = ['{}_{}'.format(x.split('_')[0], x.split('_')[1]) for x in self.slave_dataset.index.values]
        else:
            self.groups = None

    def hyperparameter_tuning(self):
        """ Comment """
        # gs = GridSearchCV(self.machina, self.machina_tuning_parameters, cv=10, return_train_score=True)
        gs = RandomizedSearchCV(self.machina, self.machina_tuning_parameters, cv=GroupKFold(10),
                                return_train_score=True, n_iter=500)
        gs.fit(self.features, self.labels, groups=self.groups)
        pd.DataFrame(gs.cv_results_).to_csv('{}\\p-tune_{}.csv'.format(self.svfl, self.svnm))

    def set_learner(self, learner):
        """ Comment """
        if learner == 'randomforest':
            # v1: n_est=50, max_depth=None, minleaf=2, minsplit=2
            # v2: {'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}
            # v3: {'n_estimators': 250, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
            self.machina = RandomForestRegressor(
                n_estimators=50,
                max_depth=None,
                min_samples_leaf=2,
                min_samples_split=2,
                max_features='auto',
                bootstrap=True)

            self.machina_tuning_parameters = {
                'n_estimators': [10, 50, 100, 250, 500, 1000, 2000, 5000],
                'max_features': ['auto', 'sqrt', 10],
                'max_depth': [None, 3, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap':[True, False]
            }

        elif learner == 'adaboost':
            self.machina = AdaBoostRegressor()

        elif learner == 'tree':
            self.machina = tree.DecisionTreeRegressor()

        elif learner == 'SGD':
            pass

        elif learner == 'neuralnet':
            self.machina = MLPRegressor()

        elif learner == 'svm':
            self.machina = SVR()

        else:
            print('Learner Selection Out of Bounds. \n '
                  'Please Select a Valid Learner: randomforest, adaboost, SGD, neuralnet, svm')
            exit()

    def train_data(self):
        """ Comment """
        self.machina = self.machina.fit(self.features, self.labels)

    def predict_testdata(self):
        # Predict Activities
        data = self.test_dataset.drop(labels=['Measured Activity', 'Element Dictionary'], axis=1).values
        predvals = self.machina.predict(data)

        original_test_df = self.master_dataset.loc[self.test_dataset.index].copy()
        measvals = original_test_df.loc[:, 'Measured Activity'].values

        comparison_df = pd.DataFrame([predvals, measvals],
                           index=['Predicted Activity','Measured Activity'],
                           columns=original_test_df.index).T

        comparison_df.to_csv('.\\Results\\Predictions\\ss3-7_predict_ss8.csv')
        comparison_df.plot(x='Predicted Activity', y='Measured Activity', kind='scatter')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()

        exit()

    def predict_tenfold(self, n_validations=10, n_folds=10, sv=None):
        """ Comment """
        cv = ShuffleSplit()

        scoredf = None
        for ii in range(1, n_validations):
            score = cross_val_score(self.machina, self.features, self.labels, cv=cv)
            if scoredf is None:
                scoredf = pd.DataFrame(score)
            else:
                tempdf = pd.DataFrame(score)
                scoredf = pd.concat([scoredf, tempdf], axis=1)

        scoredf.to_csv('{}\\tenfold_{}.csv'.format(self.svfl, self.svnm))

    def predict_crossvalidate(self):
        """ Comment """
        self.predictions = cross_val_predict(self.machina, self.features, self.labels,
                                             groups=self.groups, cv=GroupKFold(10))

    def save_predictions(self):
        """ Comment """
        if self.predictions is not None:
            df = pd.DataFrame(np.array([self.slave_dataset.index, self.predictions, self.labels]).T,
                              columns=['ID', 'Predicted Activity', 'Measured Activity'])
            df.to_csv('{}\predictions-{}.csv'.format(self.svfl, self.svnm))
        else:
            print('No predictions to save...')

    def plot_predictions_basic(self, avg=False):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        tempertures = self.slave_dataset.loc[:, 'temperature'].values

        unique_temps = np.unique(tempertures)
        n_temps = len(unique_temps)
        max_temp = np.max(unique_temps)
        min_temp = np.min(unique_temps)

        if max_temp == min_temp:
            clr = 'r'

            mask = [int(max_temp) == np.array(tempertures, dtype=int)]

            plt.scatter(self.predictions[mask],
                        self.labels[mask],
                        color=clr,
                        edgecolors='k',
                        label='{} C'.format(int(max_temp)))
        else:
            pal = pals.viridis(n_temps + 1)
            clr = [str(pal[i]) for i in [int(n_temps * (float(x) - min_temp) / (max_temp - min_temp))
                                     for x in tempertures]]

            for temp in unique_temps:
                mask = [int(temp) == np.array(tempertures, dtype=int)]

                plt.scatter(self.predictions[mask],
                            self.labels[mask],
                            color=np.array(clr)[mask],
                            edgecolors='k',
                            label='{} C'.format(int(temp)))

        # if avg:
        #     final_df = pd.DataFrame()
        #
        #     for nm in unique_names:
        #         nmdf = df.loc[df.loc[:, 'Name'] == nm]
        #         unique_temp = np.unique(nmdf.loc[:, 'Temperature'].values)
        #
        #         for temperature in unique_temp:
        #             tdf = nmdf.loc[nmdf.loc[:, 'Temperature'] == temperature]
        #             add_df = tdf.iloc[0, :].copy()
        #             add_df['Measured'] = tdf['Measured'].mean()
        #             add_df['Measured Standard Error'] = tdf['Measured'].sem()
        #             add_df['Upper'] = tdf['Measured'].mean() + tdf['Measured'].sem()
        #             add_df['Lower'] = tdf['Measured'].mean() - tdf['Measured'].sem()
        #             add_df['n Samples'] = tdf['Measured'].count()
        #
        #             final_df = pd.concat([final_df, add_df], axis=1)
        #
        #     df = final_df.transpose()

        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.xlim(0,1)
        plt.ylim(0,1)
        # plt.title(self.svnm)
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}//{}.png'.format(self.svfl, self.svnm), dpi=400)
        plt.close()

    def plot_predictions(self, svnm='ML_Statistics'):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        df = pd.DataFrame(np.array([
            [int(nm.split('_')[0]) for nm in self.slave_dataset.index.values],
            self.predictions,
            self.labels,
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
            pal = pals.plasma(unique_temps+1)
            df['color'] = [pal[i]
                           for i in [int(unique_temps*(float(x)-min_temp)/(max_temp-min_temp))
                                     for x in df['Temperature']]]

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Activity"
        p.yaxis.axis_label = "Measured Activity"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(df)

        p.circle("Predicted", "Measured", size=12, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        output_file("{}\\{}.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def plot_averaged(self, whiskers=False):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        df = pd.DataFrame(np.array([
            [int(nm.split('_')[0]) for nm in self.slave_dataset.index.values],
            self.predictions,
            self.labels,
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
        p.xaxis.axis_label = "Predicted Activity"
        p.yaxis.axis_label = "Measured Activity"
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

    def extract_important_features(self, svnm=None):
        """ Save all feature importance, print top 10 """

        df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns,
                          columns=['Feature Importance'])

        print(df.sort_values(by='Feature Importance', ascending=False).head(10))

        if svnm is None:
            return df
        else:
            df.to_csv('{}//feature_importance-{}.csv'.format(self.svfl, self.svnm))


    def evaluate_learner(self):
        """ Comment """
        mask = self.labels != 0
        err = abs(np.array(self.predictions[mask]) - np.array(self.labels[mask]))
        mean_ave_err = np.mean(err / np.array(self.labels[mask]))
        acc = 1 - mean_ave_err

        mean_abs_err = mean_absolute_error(self.labels, self.predictions)
        r2 = r2_score(self.labels, self.predictions)

        print('\n----- Model {} -----'.format(self.svnm))
        print('R2: {:0.3f}'.format(r2))
        print('Average Error: {:0.3f}'.format(mean_abs_err))
        print('Accuracy: {:0.3f}'.format(acc))
        print('Time to Complete: {:0.1f} s'.format(time.time() - self.start_time))
        print('\n')

        pd.DataFrame([r2, mean_abs_err, acc, time.time() - self.start_time],
                     index=['R2','Mean Abs Error','Accuracy','Time']).to_csv('{}\\{}-eval.csv'.format(self.svfl, self.svnm))

    def plot_important_features(self):
        """ Comment """
        featdf = self.extract_important_features()
        top5feats = featdf.nlargest(5, 'Feature Importance').index.values.tolist()
        feats = self.slave_dataset.loc[:, top5feats+['Measured Activity']]
        feats['hue'] = np.ceil(feats['Measured Activity'].values * 5)

        # feats = feats[feats['temperature'] == 300.0]

        # sns.pairplot(feats, hue='temperature', y_vars=['Measured Activity'], x_vars=top5feats)
        sns.pairplot(feats, hue='temperature', diag_kind='kde')
        plt.tight_layout()
        plt.savefig('{}\\{}-featrels.png'.format(self.svfl, self.svnm))
        plt.close()

    def plot_important_features_bokeh(self, temp_slice=None, xaxis='Measured', yaxis='Predicted',
                                      xlabel='Measured Activity', ylabel='Predicted Activity',
                                      svtag='', yrng=None, xrng=None):
        """ Comment """

        featdf = pd.DataFrame(self.slave_dataset.copy())
        featdf['Name'] = [''.join('{}({})'.format(key, str(int(val))) for key, val in x)
                         for x in self.slave_dataset.loc[:, 'Element Dictionary']]
        featdf['ID'] = [x.split('_')[0] for x in featdf.index.values]
        featdf.drop(columns='Element Dictionary', inplace=True)
        featdf['Predicted'] = self.predictions
        featdf['Measured'] = self.labels
        unique_temps = len(featdf['temperature'].unique())
        max_temp = featdf['temperature'].max()
        min_temp = featdf['temperature'].min()

        if max_temp == min_temp:
            featdf['color'] = pals.plasma(5)[4]
        else:
            pal = pals.plasma(unique_temps + 1)
            featdf['color'] = [pal[i]
                           for i in [int(unique_temps * (float(x) - min_temp) / (max_temp - min_temp))
                                     for x in featdf['temperature']]]

        if temp_slice is not None:
            featdf = featdf[featdf['temperature'] == temp_slice]

        if xrng is None:
            xrng = DataRange1d()
        if yrng is None:
            yrng = DataRange1d()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@temperature')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = xrng
        p.y_range = yrng
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(featdf)

        p.circle(xaxis, yaxis, size=12, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        output_file("{}\\{}-{}.html".format(self.svfl, self.svnm, svtag), title="stats.py")
        save(p)


class Catalyst():
    """Catalyst will contain each individual training set"""
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

    def feature_add_elemental_properties(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/Elements_Cleaned.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements.keys())]

        for feature_name, feature_values in eledf.T.iterrows():
            for index, _ in enumerate(self.elements):
                self.feature_add('{nm}_{index}'.format(nm=feature_name, index=index),
                                 feature_values.values[index])


    def feature_add_n_elements(self):
        n_eles = 0
        for val in self.elements.values():
            if val > 0:
                n_eles += 1

        self.feature_add('n_elements',n_eles)


def temp_step(svfl='.\\Results\\', svnm='test'):
    skynet = Learner()
    skynet.set_learner(learner='randomforest')
    skynet.load_nh3_catalysts()
    skynet.create_master_dataset()

    for temp in [250, 300, 350, 400, 450]:
        svnm_temp = '{s}-{t}C'.format(s=svnm, t=temp)
        skynet.filter_master_dataset(filter='3ele', temperature=temp, svfl=svfl, svnm=svnm_temp)
        skynet.train_data()
        skynet.extract_important_features()
        skynet.predict_crossvalidate()
        skynet.visualize_tree()
        skynet.plot_predictions_basic()
        skynet.plot_predictions()
        skynet.plot_averaged()


if __name__ == '__main__':
    # Properties
    nm = 'v7_stats'
    dattype = 'avg' # avg or all
    filter = 'All' # 3ele, bi, or mono
    temp = None
    group = 'blind'

    # Create Names
    svfl = '.\\Results\\{}'.format('{}-avg'.format(nm) if dattype == 'avg' else nm)
    svnm = '{a}{avg}-{b}-{c}{d}'.format(
        a=filter,
        avg='-avg' if dattype=='avg' else '',
        b=nm,
        c=group,
        d='-{}'.format(temp) if temp is not None else ''
    )
    print(svnm)

    # Begin Machine Learning
    skynet = Learner(
        import_file='AllData_Condensed' if dattype=='avg' else 'AllData',
        svfl=svfl,
        svnm='{}'.format(svnm)
    )

    skynet.set_learner(learner='randomforest')
    skynet.load_nh3_catalysts()
    skynet.filter_master_dataset(filter=filter, temperature=temp, group=group)
    # skynet.hyperparameter_tuning()
    # exit()

    skynet.create_test_dataset(catids=[65,66,67,68,69,73,74,75,76,77,78,82,83])
    skynet.train_data()
    skynet.predict_testdata()
    exit()
    skynet.extract_important_features()
    skynet.predict_crossvalidate()
    skynet.evaluate_learner()

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
    #             yaxis='Measured', ylabel='Measured Activity', yrng=Range1d(0,1)
    #         )
    # skynet.save_predictions()


    # skynet.visualize_tree(svnm='{}-tree'.format(sv))
    skynet.plot_predictions()
    skynet.plot_averaged(whiskers=True)
    skynet.plot_predictions_basic()

