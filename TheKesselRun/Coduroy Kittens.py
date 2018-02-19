# Because why the hell not

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

class Learner():
    """Learner will use catalysts to construct feature-label set and perform machine learning"""

    def __init__(self):
        self.catalyst_dictionary = dict()

    def add_catalyst(self, index, catalyst):
        self.catalyst_dictionary[index] = catalyst

class Catalyst():
    """Catalyst will contain each individual training set"""
    def __init__(self):
        self.ID = None
        self.label = None
        self.features = None
        self.temperature = None
        self.elements = dict()

    def add_element(self, element, weight_loading):
        self.elements[element] = weight_loading

    def extract_element_features(self, ele):
        """ Extract the features for a single material from Elements.csv"""

        eledf = pd.read_csv('ELements.csv', index_col=1)
        feature_names = eledf.loc[ele].index.values
        features = eledf.loc[ele].values
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

        features_dict = {
            # Stochimetric Features
            'n_ele': [len(self.elements)],

            # Statistics
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

        feature_list = list()
        for feat in features_dict:
            val = features_dict[feat]
            feature_list += val

        if set:
            initial_list = list(self.elements.keys())  + list(self.elements.values())
            self.features = initial_list + [self.temperature] + feature_list
        else:
            return feature_list

    def complete(self):
        if (self.features is not None) & \
                (self.label is not None) & \
                (self.temperature is not None) & \
                (self.ID is not None):
            return True
        else:
            return False


if __name__ == '__main__':
    pass

    # Spool up Learner Class
    skynet = Learner()

    # Create Catalysts
    df = pd.read_excel(r"C:\Users\quick\Desktop\NH3Data.xlsx", index_col=0)

    for vals in df.iterrows():
        cat = Catalyst()
        cat.ID = vals[0]
        params = vals[1]
        cat.add_element(params['Metal 1'],params['M1 Wt %'])
        cat.add_element(params['Metal 2'],params['M2 Wt %'])
        cat.add_element(params['Metal 3'],params['M3 Wt %'])

        for temp in [250, 300, 350, 400, 450]:
            cat.temperature = temp
            cat.label = params['T'+str(temp)]
            cat.create_element_feature_list(set=True)
            if cat.complete():
                print('Catalyst {}.{}C complete'.format(cat.ID, cat.temperature))
                skynet.add_catalyst(index=cat.ID, catalyst=cat)
            else:
                print('Catalyst {} NOT complete.  Not added to skynet.'.format(cat.ID))




