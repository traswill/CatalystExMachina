# Because why the hell not

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure, show, output_file
import bokeh.palettes as pals
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from bokeh.models import Range1d

class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self):
        self.catalyst_dictionary = dict()

        self.feature_names = list()
        self.features = list()
        self.labels = list()
        self.predictions = None

        self.learner = None

    def add_catalyst(self, index, catalyst):
        base_index = index
        mod = 0

        # Determine if key already exists in dictionary, modify key if so
        while index in self.catalyst_dictionary:
            mod += 1
            index = '{}_{}'.format(base_index, mod)

        # Add to dictionary
        self.catalyst_dictionary[index] = catalyst

    def get_training_data_size(self):
        print(len(self.features))

    def create_training_set(self):
        for key, value in self.catalyst_dictionary.items():
            catalyst_feature_list = list()
            catalyst_feature_list += list(value.elements.values())
            catalyst_feature_list += list(value.input_dict.values())
            catalyst_feature_list += list(value.feature_dict.values())

            self.features += [catalyst_feature_list]
            self.labels += [value.activity]

        self.feature_names += ['Loading_1','Loading_2','Loading_3']
        self.feature_names += list(list(self.catalyst_dictionary.values())[0].input_dict.keys())
        self.feature_names += list(list(self.catalyst_dictionary.values())[0].feature_dict.keys())

        self.features = pd.DataFrame(self.features, dtype=float).values

    def clean_data(self):
        """Replaces NaN values with mean value of column"""
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.features = imp.fit_transform(self.features)

    def scale_data(self):
        self.features = preprocessing.minmax_scale(self.features)

    def hyperparameter_tuning(self):
        parameters = {
            'n_estimators': [10, 50, 100, 500],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3],
            'bootstrap': [True, False]
        }
        clf = GridSearchCV(self.learner, parameters)
        clf.fit(self.features, self.labels)
        pd.DataFrame(clf.cv_results_).to_csv(r'.\Results\parameter_tuning.csv')


    def set_learner(self, learner):
        if (learner == 'random forest') or (learner == 0):
            self.learner = RandomForestRegressor(n_estimators=200)
            # self.learner = RandomForestRegressor(n_estimators=100, min_sample_leaf=2, min_sample_split=6, bootstrap=True)
        elif (learner == 'adaboost') or (learner == 1):
            self.learner = AdaBoostRegressor()
        elif (learner == 'SGD') or (learner == 2):
            pass
        elif (learner == 'neural net') or (learner == 3):
            # self.learner = MLPRegressor()
            self.learner = MLPRegressor(activation='relu', learning_rate='adaptive', max_iter=1000)
        else:
            print('Learner Selection Out of Bounds.  Please Select a Valid Learner.')
            exit()

    def train_learner(self):
        self.learner.fit(self.features, self.labels)

    def validate_learner(self, n_validations=10, n_folds=10, sv=False):
        cv = ShuffleSplit()

        scoredf = None
        for ii in range(1, n_validations):
            score = cross_val_score(self.learner, self.features, self.labels, cv=cv)
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
        self.predictions = cross_val_predict(self.learner, self.features, self.labels, cv=10)

    def plot_predictions_basic(self):
        temperature = [self.catalyst_dictionary[x].input_dict['temperature'] for x in self.catalyst_dictionary]
        plt.scatter(self.labels, self.predictions, c=temperature)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()

    def plot_predictions(self, sv_nm='ML_Statistics'):
        df = pd.DataFrame(np.array([
            [self.catalyst_dictionary[x].ID for x in self.catalyst_dictionary],
            self.predictions,
            self.labels,
            [self.catalyst_dictionary[x].input_dict['temperature'] for x in self.catalyst_dictionary]]).T,
                          columns=['ID', 'Predicted', 'Measured', 'Temperature'])

        cat_eles = [self.catalyst_dictionary[x].elements for x in self.catalyst_dictionary]
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

        pal = pals.plasma(5)
        df['color'] = [pal[i] for i in [int(4*(x-250)/(450-250)) for x in df['Temperature']]]

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

        output_file(".\\Figures\\{}.html".format(sv_nm), title="stats.py")
        show(p)

    def load_NH3_catalysts(self, filter_monometallics=False, filter_bimetallics=False):
        df = pd.read_excel(r".\Data\NH3Data.xlsx", index_col=0)

        for vals in df.iterrows():
            if np.isnan(vals[0]):
                continue

            for temp in [250, 300, 350, 400, 450]:
                cat = Catalyst()
                cat.ID = int(vals[0])
                inputs = vals[1]

                if np.isnan(inputs['T' + str(temp)]):
                    continue

                if filter_monometallics & (inputs['Element_2'] == 'None') & (inputs['Element_3'] == 'None_2'):
                    continue

                if filter_bimetallics & ((inputs['Element_2'] == 'None') & (inputs['Element_3'] != 'None_2') |
                                                 (inputs['Element_2'] != 'None') & (inputs['Element_3'] == 'None_2')):
                    continue

                cat.add_element(inputs['Element_1'], inputs['Loading_1'])
                cat.add_element(inputs['Element_2'], inputs['Loading_2'])
                cat.add_element(inputs['Element_3'], inputs['Loading_3'])

                cat.input_reactor_number(inputs['Reactor'])
                cat.input_temperature(temp)
                cat.input_space_velocity(inputs['Space Velocity'])
                cat.input_ammonia_concentration(inputs['Ammonia Concentration'])

                cat.activity = inputs['T' + str(temp)]

                cat.feature_add_n_elements()
                cat.feature_add_elemental_properties()

                self.add_catalyst(index='{ID}_{T}'.format(ID=cat.ID, T=temp), catalyst=cat)

    def save_predictions(self):
        if self.predictions is not None:
            df = pd.DataFrame(np.array([['{ID}_{T}'.format(ID=self.catalyst_dictionary[x].ID,
                                                           T=self.catalyst_dictionary[x].input_dict['temperature'])
                                         for x in skynet.catalyst_dictionary], self.predictions, self.labels]).T,
                              columns=['ID', 'Predicted Activity', 'Measured Activity'])
            df.to_csv(r'.\Results\predictions.csv')


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




if __name__ == '__main__':
    skynet = Learner()
    skynet.load_NH3_catalysts(filter_bimetallics=True, filter_monometallics=True)
    skynet.create_training_set()
    skynet.clean_data()
    # skynet.scale_data()
    skynet.set_learner(learner=0)

    # skynet.hyperparameter_tuning()
    # skynet.validate_learner(sv=True)
    skynet.predict_learner()
    skynet.save_predictions()
    # skynet.plot_predictions_basic()
    skynet.plot_predictions(sv_nm='NH3_NoBimetallicsNoMonometallics')

