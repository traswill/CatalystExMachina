# Because why the hell not

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, cross_val_predict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure, show, output_file
import bokeh.palettes as pals

class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self):
        self.catalyst_dictionary = dict()

        self.feature_names = list()
        self.features = list()
        self.filtered_features = list()
        self.labels = list()
        self.predictions = None

        self.learner = None

    def add_catalyst(self, index, catalyst):
        self.catalyst_dictionary[index] = catalyst

    def get_training_data_size(self):
        print(len(self.filtered_features))

    def create_training_set(self):
        feature_df = None
        feature_names = None
        label_list = list()

        for val in self.catalyst_dictionary.values():
            temp_df = pd.DataFrame(val.features)
            label_list += [val.label]

            if feature_df is None:
                feature_df = pd.DataFrame(val.features)
                feature_names = val.feature_names
            else:
                feature_df = pd.concat([feature_df, temp_df], axis=1)

        feature_df = feature_df.fillna(0)
        self.features = feature_df.values[3:].T
        self.feature_names = feature_names[3:]
        self.filtered_features = self.features
        self.labels = label_list

    def reset_features(self):
        self.filtered_features = self.features

    def trim_features(self):
        print(self.filtered_features)

    def set_learner(self, learner):
        if (learner == 'random forest') or (learner == 0):
            self.learner = RandomForestRegressor(n_estimators=200)
        else:
            print('Learner Selection Out of Bounds.  Please Select a Valid Learner.')
            exit()

    def train_learner(self):
        self.learner.fit(self.filtered_features, self.labels)

    def validate_learner(self, n_validations=10, n_folds=10, sv=False):
        cv = ShuffleSplit()

        scoredf = None
        for ii in range(1, n_validations):
            score = cross_val_score(self.learner, self.filtered_features, self.labels, cv=cv)
            if scoredf is None:
                scoredf = pd.DataFrame(score)
            else:
                tempdf = pd.DataFrame(score)
                scoredf = pd.concat([scoredf, tempdf], axis=1)

        if sv:
            scoredf.to_csv(r'C:\Users\quick\Desktop\results.csv')
        else:
            return scoredf

    def predict_learner(self):
        self.predictions = cross_val_predict(self.learner, self.features, self.labels, cv=10)

    def compare_predictions(self):
        temps = [self.catalyst_dictionary[x].temperature for x in self.catalyst_dictionary]
        plt.scatter(self.labels, self.predictions, c=temps)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()

    def bokeh_plot(self):
        df = pd.DataFrame(np.array([
            [self.catalyst_dictionary[x].ID for x in self.catalyst_dictionary],
            self.predictions,
            self.labels,
            [self.catalyst_dictionary[x].temperature for x in self.catalyst_dictionary]]).T,
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

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=1200, title='')
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Activity"
        p.yaxis.axis_label = "Measured Activity"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(df)

        p.circle("Predicted", "Measured", size=12, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        output_file("ML_Statistics.html", title="stats.py")

        show(p)

class Catalyst():
    """Catalyst will contain each individual training set"""
    def __init__(self):
        self.ID = None
        self.label = None

        self.features = None
        self.feature_names = None
        self.temperature = None

        self.reactor_index = None
        self.space_velocity = None

        self.input_dict = dict()
        self.feature_dict = dict()
        self.elements = dict()

    def input_temperature(self, T):
        self.input_dict['temperature'] = T

    def input_elements(self):
        self.input_dict['elements'] = self.elements

    def input_space_velocity(self, space_velocity):
        self.input_dict['space_velocity'] = space_velocity

    def input_reactor_number(self, reactor_number):
        self.input_dict['reactor_number'] = reactor_number

    def add_element(self, element, weight_loading):
        self.elements[element] = weight_loading

    def extract_element_features(self, ele):
        """ Extract the features for a single material from Elements.csv"""

        eledf = pd.read_csv(r'./Data/ELements.csv', index_col=1)
        feature_names = eledf.loc[str(ele)].index.values
        features = eledf.loc[str(ele)].values
        return features, feature_names

    def create_element_feature_list(self, set=False):
        """Create feature list using mean, median, etc of all elements in catalyst"""

        feature_df = None

        for ele in self.elements:
            if ele is np.nan:
                pass
            else:
                feat, feat_nm = self.extract_element_features(ele)
                if feature_df is None:
                    feature_df = pd.DataFrame(feat, columns=[ele], index=feat_nm)
                else:
                    temp_df = pd.DataFrame(feat, columns=[ele], index=feat_nm)
                    feature_df = pd.concat([feature_df, temp_df], axis=1)

        # Extract Statistic Properties from Elements.csv values
        def extract(df, property):
            # Extract max, mean, median, min for all proprties
            return [
                df.loc[property, :].max(),
                df.loc[property, :].mean(),
                df.loc[property, :].median(),
                df.loc[property, :].min(),
            ]

        # Stoichiometric Features
        stoich_features_dict = {
            'n_ele': [len(self.elements)],
        }

        stoich_feature_list = list()
        stoich_feature_nm_list = list()
        for feat in stoich_features_dict:
            stoich_feature_list += stoich_features_dict[feat]
            stoich_feature_nm_list += [feat]

        # Statistical Features (calculates max, mean, median, and min)
        stat_features_dict = {
            'atvol': extract(feature_df, 'AtomicVolume'),
            'atwt': extract(feature_df, 'AtomicWeight'),
            'atrad': extract(feature_df, 'CovalentRadius'),
            'bT': extract(feature_df, 'BoilingTemp'),
            'dens': extract(feature_df, 'Density'),
            'eaff': extract(feature_df, 'ElectronAffinity'),
            'eneg': extract(feature_df, 'Electronegativity'),
            'mT': extract(feature_df, 'MeltingT'),
            'f_unf': extract(feature_df, 'NfUnfilled'),
            'f_val': extract(feature_df, 'NfValence'),
            'd_unf': extract(feature_df, 'NdUnfilled'),
            'd_val': extract(feature_df, 'NdValence'),
            'p_unf': extract(feature_df, 'NpUnfilled'),
            'p_val': extract(feature_df, 'NpValence'),
            's_unf': extract(feature_df, 'NsUnfilled'),
            's_val': extract(feature_df, 'NsValence'),
            'n_unf': extract(feature_df, 'NUnfilled'),
            'n_val': extract(feature_df, 'NValence')
        }

        stat_feature_list = list()
        stat_feature_nm_list = list()
        for feat in stat_features_dict:
            stat_feature_list += stat_features_dict[feat]
            stat_feature_nm_list += [
                '{nm}_max'.format(nm=feat),
                '{nm}_mean'.format(nm=feat),
                '{nm}_median'.format(nm=feat),
                '{nm}_min'.format(nm=feat)]

        # Add all features and names to a list
        features = list(self.elements.keys()) + list(self.elements.values())
        feature_names = ['M1','M2','M3','W1','W2','W3']

        features += [self.temperature] + [self.reactor_index] + [self.space_velocity]
        feature_names += ['T','Reactor','Space Velocity']

        features += stoich_feature_list
        feature_names += stoich_feature_nm_list

        features += stat_feature_list
        feature_names += stat_feature_nm_list

        if set:
            self.features = features
            self.feature_names = feature_names
        else:

            return features, feature_names


if __name__ == '__main__':
    # Spool up Learner Class
    skynet = Learner()

    # Create Catalysts
    df = pd.read_excel(r".\Data\NH3Data.xlsx", index_col=0)

    for vals in df.iterrows():
        for temp in [250, 300, 350, 400, 450]:
            cat = Catalyst()
            cat.ID = vals[0]
            params = vals[1]
            cat.add_element(params['Metal 1'],params['M1 Wt %'])
            cat.add_element(params['Metal 2'],params['M2 Wt %'])
            cat.add_element(params['Metal 3'],params['M3 Wt %'])
            cat.reactor_index = params['Reactor']
            cat.temperature = temp
            cat.label = params['T'+str(temp)]
            cat.space_velocity = params['Space Velocity']
            if np.isnan(cat.label):
                continue
            cat.create_element_feature_list(set=True)
            if cat.complete():
                # Prevent overwriting of duplicate keys
                keymod = 0
                catkey = '{ID}_{T}_{mod}'.format(ID=cat.ID, T=cat.temperature, mod=keymod)
                while catkey in skynet.catalyst_dictionary:
                    keymod+=1
                    catkey = '{ID}_{T}_{mod}'.format(ID=cat.ID, T=cat.temperature, mod=keymod)

                # Once catalyst has a unique ID modifier, add to skynet
                skynet.add_catalyst(index=catkey, catalyst=cat)
            else:
                print('Catalyst {} NOT complete.  Not added to skynet.'.format(cat.ID))

    # Create training data
    skynet.create_training_set()
    skynet.set_learner(learner=0)
    # skynet.validate_learner(sv=True)
    skynet.predict_learner()
    # skynet.compare_predictions()
    skynet.bokeh_plot()
    exit()


    df = pd.DataFrame(np.array([['{ID}_{T}'.format(ID=skynet.catalyst_dictionary[x].ID,
                                                   T=skynet.catalyst_dictionary[x].temperature)
                                 for x in skynet.catalyst_dictionary], skynet.predictions, skynet.labels]).T,
                      columns=['ID','Predicted Activity','Measured Activity'])
    df.to_csv(r'C:\Users\quick\Desktop\predictions.csv')
