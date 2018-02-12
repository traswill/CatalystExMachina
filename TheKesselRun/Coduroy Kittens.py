# Because why the hell not

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class Catalyst():
    def __init__(self):
        self.labels = None
        self.features = None
        self.metals = None

    def set(self, labels=None, features=None):
        self.labels = labels
        self.features = features

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
        df = None

        for ele in self.metals:
            feat, feat_nm = self.extract_element_features(ele)
            if df is None:
                df = pd.DataFrame(feat, index=ele, columns=feat_nm)
            else:
                df = df.append(feat, index=ele)




    def to_fit(self):
        """ Convert self.labels and self.features into formatted data for quick insertion into sklearn's XX.fit command
        :return:
        """
        pass

if __name__ == '__main__':
    pass
    cat = Catalyst()
    f, fn = cat.extract_element_features(ele='H')
    print(f)



