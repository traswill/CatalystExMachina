# Because why the hell not

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class Catalyst():
    def __init__(self):
        self.labels = None
        self.features = None
        self.elements = dict()

    def add_element(self, element, weight_loading):
        self.elements[element] = weight_loading

    def modify_features(self):
        """ Extract relevant information from a lookup table for elements
        :return:
        """

    def extract_element_features(self, ele):
        """ Extract the features for a single material from Elements.csv
        :param ele: Material Abbreviation as a string
        :return features: The values associated with the specified element
        """

        eledf = pd.read_csv('ELements.csv', index_col=1)
        feature_names = eledf.loc[ele].index.values
        features = eledf.loc[ele].values
        return features, feature_names

    def create_element_feature_list(self):
        """Create feature list using mean, median, etc of all elements in catalyst"""

        feature_df = None

        for ele in self.elements:
            feat, feat_nm = self.extract_element_features(ele)
            if feature_df is None:
                feature_df = pd.DataFrame(feat, columns=[ele], index=feat_nm)
            else:
                temp_df = pd.DataFrame(feat, columns=[ele], index=feat_nm)
                feature_df = pd.concat([feature_df, temp_df], axis=1)

        # Elemental Property (Statistic) Features
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
            'n_ele': len(self.elements),

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

        print(features_dict)

    def to_fit(self):
        """ Convert self.labels and self.features into formatted data for quick insertion into sklearn's XX.fit command
        :return:
        """
        pass

if __name__ == '__main__':
    pass
    cat = Catalyst()
    cat.add_element('Co', 3)
    cat.add_element('Pt', 3)
    cat.add_element('K', 3)

    cat.create_element_feature_list()



