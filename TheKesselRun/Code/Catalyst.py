import pandas as pd
import numpy as np


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

    def input_standard_error(self, error):
        self.input_dict['standard error'] = error

    def input_n_averaged_samples(self, n_avg):
        self.input_dict['n_averaged'] = n_avg

    def input_n_Cl_atoms(self, Cl_atoms):
        self.input_dict['n_Cl_atoms'] = Cl_atoms

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
        eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
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
            'Conductivity': calc_weighted_average,
            'Covalent Radius': calc_weighted_average,
            'Density': calc_weighted_average,
            'Dipole Polarizability': calc_weighted_average,
            'Electron Affinity': calc_weighted_average,
            'Electronegativity': calc_weighted_average,
            'Fusion Enthalpy': calc_weighted_average,
            'GS Bandgap': calc_weighted_average,
            'GS Energy': calc_weighted_average,
            'Heat Capacity (Mass)': calc_weighted_average,
            'Heat Capacity (Molar)': calc_weighted_average,
            'Heat Fusion': calc_weighted_average,
            'First Ionization Energy': calc_weighted_average,
            'Second Ionization Energy': calc_weighted_average,
            'Third Ionization Energy': calc_weighted_average,
            'Fourth Ionization Energy': calc_weighted_average,
            'Fifth Ionization Energy': calc_weighted_average,
            'Sixth Ionization Energy': calc_weighted_average,
            'Seventh Ionization Energy': calc_weighted_average,
            'Eighth Ionization Energy': calc_weighted_average,
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
            'Polarizability': calc_weighted_average,
            'Periodic Table Row': calc_statistics,
            'phi': calc_statistics,
            'Zunger Pseudopotential (d)': calc_statistics,
            'Zunger Pseudopotential (p)': calc_statistics,
            'Zunger Pseudopotential (pi)': calc_statistics,
            'Zunger Pseudopotential (s)': calc_statistics,
            'Zunger Pseudopotential (sigma)': calc_statistics,
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
