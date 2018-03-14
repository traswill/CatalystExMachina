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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker
from bokeh.plotting import figure, show, output_file
import bokeh.palettes as pals
from bokeh.models import Range1d

import seaborn as sns
import ast


class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self):
        self.catalyst_dictionary = dict()

        self.master_dataset = pd.DataFrame()
        self.slave_dataset = pd.DataFrame()
        self.features_df = pd.DataFrame()
        self.plot_df = pd.DataFrame()

        self.features = list()
        self.labels = list()
        self.predictions = list()

        self.machina = None
        self.machina_tuning_parameters = None

        self.filter = None

    def add_catalyst(self, index, catalyst):
        base_index = index
        mod = 0

        # Determine if key already exists in dictionary, modify key if so
        while index in self.catalyst_dictionary:
            mod += 1
            index = '{}_{}'.format(base_index, mod)

        # Add to dictionary
        self.catalyst_dictionary[index] = catalyst

    def load_nh3_catalysts(self):
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
            cat.activity = row['Concentration']

            cat.feature_add_n_elements()
            cat.feature_add_elemental_properties()

            self.add_catalyst(index='{ID}_{T}'.format(ID=cat.ID, T=row['Temperature']), catalyst=cat)

    def create_training_set(self):
        # Set up catalyst loading dictionary with loadings
        loading_df = pd.read_csv('.\\Data\\Elements.csv', usecols=['Abbreviation'], index_col='Abbreviation').transpose()
        loading_df.columns = ['{} Loading'.format(ele) for ele in loading_df.columns]

        i = 0
        for catid, catobj in self.catalyst_dictionary.items():
            # Reset loading dictionary
            load_df = loading_df.copy()

            # Add elements and loading to loading dict
            for ele, wt in catobj.elements.items():
                load_df.loc[catid, '{} Loading'.format(ele)] = wt

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
        self.master_dataset.fillna(value=-1, inplace=True)

    def filter_training_set(self, filter=None, temperature=None, features=None):
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

        self.filter = np.union1d(filter_elements(filter),
                                 filter_temp(temperature)
                                 )

        # Filter Master to Slave
        self.slave_dataset = self.master_dataset.drop(index=self.filter).copy()

        # Set all other DFs from slave
        self.features_df = self.slave_dataset.drop(labels=['Measured Activity', 'Element Dictionary'], axis=1).copy()
        self.labels_df = self.slave_dataset['Measured Activity'].copy()
        self.plot_df = self.slave_dataset.loc[:, ['Measured Activity', 'Element Dictionary']].copy()

        # Set Features and Labels
        self.features = self.features_df.values
        self.labels = self.labels_df.values

    def hyperparameter_tuning(self):
        if (self.machina == 'randomforest'):
            self.machina_tuning_parameters = {
                'n_estimators': [10, 50, 100, 500],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'bootstrap': [True, False]
            }

        elif (self.machina == 'adaboost'):
            pass

        elif (self.machina == 'SGD'):
            pass

        elif (self.machina == 'neuralnet'):
            pass

        elif (self.machina == 'svm'):
            pass

        else:
            print('Learner Selection Out of Bounds.  Please Select a Valid Learner: randomforest, adaboost, neuralnet, svm')
            exit()

        clf = GridSearchCV(self.machina, self.machina_tuning_parameters)
        clf.fit(self.features, self.labels)
        pd.DataFrame(clf.cv_results_).to_csv(r'.\Results\parameter_tuning.csv')

    def set_learner(self, learner):
        if learner == 'randomforest':
            self.machina = RandomForestRegressor(min_samples_leaf=2, min_samples_split=2, n_estimators=50)

            self.machina_tuning_parameters = {
                'n_estimators': [10, 50, 100, 500],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'bootstrap': [True, False]
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

    def train_learner(self):
        self.machina.fit(self.features, self.labels)

    def validate_learner(self, n_validations=10, n_folds=10, sv=False):
        cv = ShuffleSplit()

        scoredf = None
        for ii in range(1, n_validations):
            score = cross_val_score(self.machina, self.features, self.labels, cv=cv)
            if scoredf is None:
                scoredf = pd.DataFrame(score)
            else:
                tempdf = pd.DataFrame(score)
                scoredf = pd.concat([scoredf, tempdf], axis=1)

        if sv:
            scoredf.to_csv(r'.\Results\results.csv')
        else:
            return scoredf

    def predict_learner(self):
        self.predictions = cross_val_predict(self.machina, self.features, self.labels, cv=10)

    def plot_predictions_basic(self, svnm='Fig'):
        self.predict_learner()
        plt.scatter(self.predictions, self.labels, c=self.slave_dataset.loc[:, 'temperature'], edgecolors='k')
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig('.//Figures//{}.png'.format(svnm))
        plt.close()

    def plot_predictions(self, svnm='ML_Statistics'):
        if self.predictions is None:
            self.predict_learner()

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

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title='')
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Activity"
        p.yaxis.axis_label = "Measured Activity"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(df)

        p.circle("Predicted", "Measured", size=12, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        output_file(".\\Figures\\{}.html".format(svnm), title="stats.py")
        show(p)

    def plot_averaged(self, svnm='ML_Statistics_Averaged', whiskers=False):
        if self.predictions is None:
            self.predict_learner()

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

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title='')
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

        output_file(".\\Figures\\{}.html".format(svnm), title="stats.py")
        show(p)

    def save_predictions(self):
        if self.predictions is not None:
            df = pd.DataFrame(np.array([['{ID}_{T}'.format(ID=self.catalyst_dictionary[x].ID,
                                                           T=self.catalyst_dictionary[x].input_dict['temperature'])
                                         for x in self.catalyst_dictionary], self.predictions, self.labels]).T,
                              columns=['ID', 'Predicted Activity', 'Measured Activity'])
            df.to_csv(r'.\Results\predictions.csv')
        else:
            print('No predictions to save...')

    def visualize_tree(self):
        """ Find a way to visualize the decision tree """
        pass

    def extract_important_features(self, svnm='Feature_Importance'):
        """ Save all feature importances, print top 10 """
        df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns, columns=['Feature Importance'])
        df.to_csv('.//Results//{}.csv'.format(svnm))
        print(df.sort_values(by='Feature Importance', ascending=False).head(10))


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
        self.elements[element] = weight_loading

    def input_space_velocity(self, space_velocity):
        self.input_dict['space_velocity'] = space_velocity

    def input_reactor_number(self, reactor_number):
        self.input_dict['reactor_number'] = reactor_number

    def input_ammonia_concentration(self, ammonia_concentration):
        self.input_dict['ammonia_concentration'] = ammonia_concentration

    def feature_add(self, key, value):
        self.feature_dict[key] = value

    def feature_add_statistics(self, key, values):
        values = np.array(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size: # Skip if no non-nan values
            # Maximum
            self.feature_dict['{}_max'.format(key)] = np.max(values)

            # Minimum
            self.feature_dict['{}_min'.format(key)] = np.min(values)

            # Mean
            self.feature_dict['{}_mean'.format(key)] = np.mean(values)

            # Median
            self.feature_dict['{}_med'.format(key)] = np.median(values)

    def feature_add_elemental_properties(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/ELements_Cleaned.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements.keys())]

        for feature_name, feature_values in eledf.T.iterrows():
            self.feature_add('{}_0'.format(feature_name), feature_values.values[0])
            self.feature_add('{}_1'.format(feature_name), feature_values.values[1])
            self.feature_add('{}_2'.format(feature_name), feature_values.values[2])

    def feature_add_n_elements(self):
        n_eles = 0
        for val in self.elements.values():
            if val > 0:
                n_eles += 1

        self.feature_add('n_elements',n_eles)

def temp_step():
    skynet = Learner()
    skynet.set_learner(learner='randomforest')

    for temp in [250, 300, 350, 400, 450]:
        skynet.create_training_set()
        skynet.preprocess_data(clean=True)
        skynet.train_learner()
        skynet.extract_important_features('3-Element-v2-{}C-Feature-Importance'.format(temp))
        skynet.predict_learner()
        skynet.plot_predictions_basic(svnm='3-Element-v2-{}C'.format(temp))

if __name__ == '__main__':
    # temp_step()
    # exit()
    skynet = Learner()
    skynet.set_learner(learner='randomforest')
    skynet.load_nh3_catalysts()
    skynet.create_training_set()
    skynet.filter_training_set(filter='3ele', temperature=400)

    skynet.train_learner()
    skynet.extract_important_features()
    skynet.predict_learner()
    skynet.plot_predictions(svnm='3-Element-v2-400')
    # skynet.plot_averaged(svnm='3-Element-v2-400')
    skynet.plot_predictions_basic(svnm='3-Element-v2')

