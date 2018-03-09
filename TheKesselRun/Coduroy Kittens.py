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

        self.create_training_set()

    def create_training_set(self):
        feats = list()
        feat_names = list()
        lbls = list()

        for key, value in self.catalyst_dictionary.items():
            catalyst_feature_list = list()
            catalyst_feature_list += list(value.elements.values())
            catalyst_feature_list += list(value.input_dict.values())
            catalyst_feature_list += list(value.feature_dict.values())

            feats += [catalyst_feature_list]
            lbls += [value.activity]

        feat_names += ['Loading_1','Loading_2','Loading_3']
        feat_names += list(list(self.catalyst_dictionary.values())[0].input_dict.keys())
        feat_names += list(list(self.catalyst_dictionary.values())[0].feature_dict.keys())

        self.features_df = pd.DataFrame(feats, dtype=float, columns=feat_names)
        self.labels_df = pd.DataFrame(lbls, columns=['Measured Activity'])

        other_df = pd.DataFrame(np.array([
            [self.catalyst_dictionary[x].ID for x in self.catalyst_dictionary],
            [self.catalyst_dictionary[x].elements for x in self.catalyst_dictionary],
        ]).T,
                                columns=['ID', 'Element Dictionary'])

        self.master_dataset = pd.concat([other_df, self.labels_df, self.features_df], axis=1)
        self.slave_dataset = self.master_dataset

    def filter_catalysts(self, filter=None):
        df = pd.DataFrame([list(x.keys()) for x in [cat.elements for cat in self.catalyst_dictionary.values()]],
                          columns=['Ele1','Ele2','Ele3'])

        if filter == 'mono':
            self.filter = df[(df.loc[:, 'Ele2'] == '-') & (df.loc[:, 'Ele3'] == '--')].index

        elif filter == 'bi':
            self.filter = df[((df.loc[:, 'Ele2'] == '-') & (df.loc[:, 'Ele3'] != '--')) |
                            ((df.loc[:, 'Ele2'] != '-') & (df.loc[:, 'Ele3'] == '--'))].index
        elif filter == '3ele':
            self.filter = df[(df.loc[:, 'Ele2'] == '-') | (df.loc[:, 'Ele3'] == '--')].index

        else:
            self.filter = []

        self.slave_dataset = self.master_dataset.drop(self.filter)
        self.set_dataframes()

    def filter_training_by_temperature(self):
        """ Select all of one temperature, then train data to remove temperature as a major effect """
        pass

    def set_dataframes(self):
        self.features_df = self.slave_dataset.drop(labels=['Measured Activity', 'ID', 'Element Dictionary'], axis=1)
        self.labels_df = self.slave_dataset['Measured Activity']
        self.plot_df = self.slave_dataset.loc[:, ['Measured Activity', 'ID', 'Element Dictionary']]

        self.features = self.features_df.values
        self.labels = self.labels_df.values

    def preprocess_data(self, clean=False, scale=False):
        if clean:
            # Replaces NaN values with mean value of column
            imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
            self.features = imp.fit_transform(self.features)

        if scale:
            self.features = preprocessing.minmax_scale(self.features.values)

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

    def plot_predictions(self, svnm='ML_Statistics'):
        if self.predictions is None:
            self.predict_learner()

        df = pd.DataFrame(np.array([
            self.slave_dataset.loc[:, 'ID'],
            self.predictions,
            self.labels,
            self.slave_dataset.loc[:, 'temperature']]).T,
                          columns=['ID', 'Predicted', 'Measured', 'Temperature'])

        cat_eles = self.slave_dataset.loc[:, 'Element Dictionary']
        vals = [''.join('{}({})'.format(key, str(int(val))) for key, val in x.items()) for x in cat_eles]
        vals = [strg.replace('None(0)','') for strg in vals]
        vals = [strg.replace('None_2(0)','') for strg in vals]
        df['Name'] = vals

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@Temperature')
        ])
        tools.append(hover)

        pal = pals.plasma(7)
        df['color'] = [pal[i] for i in [int(6*(x-250)/(450-250)) for x in df['Temperature']]]

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

        print(pd.DataFrame(np.array([
            self.slave_dataset.loc[:, 'ID'].values,
            self.predictions,
            self.labels,
            self.slave_dataset.loc[:, 'temperature'].values]).T))

        df = pd.DataFrame(np.array([
            self.slave_dataset.loc[:, 'ID'].values,
            self.predictions,
            self.labels,
            self.slave_dataset.loc[:, 'temperature'].values]).T,
                          columns=['ID', 'Predicted', 'Measured', 'Temperature'])

        cat_eles = self.slave_dataset.loc[:, 'Element Dictionary']
        vals = [''.join('{}({})'.format(key, str(int(val))) for key, val in x.items()) for x in cat_eles]
        df['Name'] = vals

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@Temperature')
        ])
        tools.append(hover)

        pal = pals.plasma(7)
        df['color'] = [pal[i] for i in [int(6*(x-150)/(450-150)) for x in df['Temperature']]]

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
        """ Find a way to visualize the decision treeu """
        pass

    def extract_important_features(self):
        """ Save all feature importances, print top 10 """
        df = pd.DataFrame(self.machina.feature_importances_, index=self.features_df.columns, columns=['Feature Importance'])
        df.to_csv('.//Results//Feature_Importance.csv')
        print(df.sort_values(by='Feature Importance', ascending=False).head(10))

    def restrict_feature_set(self):
        """ Allow user to specify features, then predict using only those features """
        pass


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
        eledf = pd.read_csv(r'./Data/ELements.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements.keys())]

        for feature_name, feature_values in eledf.T.iterrows():
            if feature_name == 'OxidationStates':
                continue  # Oxidation State loads as a string and causes errors.  TODO: Fix this?

            self.feature_add_statistics(feature_name, feature_values.values)

    def feature_add_n_elements(self):
        self.feature_dict['n_elements'] = len(self.elements.keys())


def generate_plots():
    skynet = Learner()
    skynet.set_learner(learner='randomforest')
    skynet.load_nh3_catalysts()

    # skynet.filter_catalysts()
    # skynet.preprocess_data(clean=True)
    # skynet.plot_predictions_basic(svnm='All-Data')
    #
    # skynet.filter_catalysts(filter='3ele')
    # skynet.preprocess_data(clean=True)
    # skynet.plot_predictions_basic(svnm='3-Element')
    #
    # skynet.filter_catalysts(filter='mono')
    # skynet.preprocess_data(clean=True)
    # skynet.plot_predictions_basic(svnm='No-Monometallics')
    #
    # skynet.filter_catalysts(filter='bi')
    # skynet.preprocess_data(clean=True)
    # skynet.plot_predictions_basic(svnm='No-Bimetallics')

    skynet.filter_catalysts(filter='3ele')
    skynet.preprocess_data(clean=True)
    skynet.train_learner()
    skynet.extract_important_features()
    skynet.predict_learner()
    skynet.plot_averaged(svnm='3-Element')


if __name__ == '__main__':
    generate_plots()

    # skynet = Learner()
    # skynet.load_nh3_catalysts_updated(filter_monometallics=False, filter_bimetallics=False)
    # skynet.preprocess_data(clean=True)
    # skynet.set_learner(learner='randomforest')
    # skynet.plot_predictions_basic()

    # skynet.plot_averaged()
    # skynet.hyperparameter_tuning()
    # skynet.validate_learner(sv=True)
    # skynet.train_learner()
    # skynet.extract_important_features()
    # skynet.save_predictions()

    # skynet.predict_learner()
    # skynet.plot_predictions()

# TODO Modify plot to incorporate experimental error
# TODO Plot only by temperature (remove it as a variable)
