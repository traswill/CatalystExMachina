# Because why the hell not

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, learning_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self):
        self.catalyst_dictionary = dict()

        self.feature_names = list()
        self.features = list()
        self.filtered_features = list()
        self.labels = list()

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
        scoredf = None
        for ii in range(1, n_validations):
            score = cross_val_score(self.learner, self.filtered_features, self.labels, cv=n_folds)
            if scoredf is None:
                scoredf = pd.DataFrame(score)
            else:
                tempdf = pd.DataFrame(score)
                scoredf = pd.concat([scoredf, tempdf], axis=1)

        if sv:
            scoredf.to_csv(r'C:\Users\quick\Desktop\results.csv')
        else:
            return scoredf

class Catalyst():
    """Catalyst will contain each individual training set"""
    def __init__(self):
        self.ID = None
        self.label = None
        self.features = None
        self.feature_names = None
        self.temperature = None
        self.elements = dict()
        self.reactor_index = None
        self.space_velocity = None

    def add_element(self, element, weight_loading):
        self.elements[element] = weight_loading

    def set_feature_temperature(self, temp):
        self.features[6] = temp

    def extract_element_features(self, ele):
        """ Extract the features for a single material from Elements.csv"""

        eledf = pd.read_csv('ELements.csv', index_col=1)
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

    def complete(self):
        if (self.features is not None) & \
                (self.label is not None) & \
                (self.temperature is not None) & \
                (self.ID is not None):
            return True
        else:
            return False


if __name__ == '__main__':
    # Spool up Learner Class
    skynet = Learner()

    # Create Catalysts
    df = pd.read_excel(r"C:\Users\quick\Desktop\NH3Data.xlsx", index_col=0)

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
    # print(skynet.filtered_features, skynet.feature_names)
    # df = pd.DataFrame(skynet.filtered_features, columns=skynet.feature_names)

    skynet.set_learner(learner=0)
    skynet.validate_learner(sv=True)



