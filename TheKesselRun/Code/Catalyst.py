# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np

class CatalystObject():
    """
    CatalystObject contains all information needed to describe a catalyst.
    It possesses methods to calculate any required features from the information provided by the user.
    """

    def __init__(self):
        """ All variables initialize to empty (None or empty dictionaries. """
        self.ID = None
        self.group = None

        self.observation_dict = dict() # dictionary of observation objects (see class below)
        self.feature_dict = dict() # dictionary of feature-value pairs
        self.elements_wt = dict() # dictionary of element-weight loading pairs
        self.elements_mol = dict() # dictionary of element-mol fraction pairs
        self.support = None

    def add_observation(self, temperature=None, space_velocity=None, gas=None, gas_concentration=None, pressure=None,
                        reactor_number=None, activity=None, selectivity=None, activity_error=None):
        """ Create and add observation object to the observation dictionary. Index increments based on number of
        entries. """
        obs = CatalystObservation()
        obs.temperature = temperature
        obs.space_velocity = space_velocity
        obs.gas = gas
        obs.concentration = gas_concentration
        obs.pressure = pressure
        obs.reactor = reactor_number
        obs.activity = activity
        obs.selectivity = selectivity
        obs.activity_error = activity_error

        self.observation_dict[len(self.observation_dict)] = obs

    def add_element(self, element, weight_loading):
        if (element != '-') & (element != '--'):
            self.elements_wt[element] = weight_loading

    def set_support(self, support):
        self.support = support

    def add_n_cl_atoms(self, cl_atoms):
        self.feature_dict['n_Cl_atoms'] = cl_atoms

    def set_group(self, group):
        self.group = group

    def feature_add(self, key, value):
        self.feature_dict[key] = value

    def add_unweighted_features(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/Elements_01Feb19.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements_wt.keys())]

        for prop in eledf:
            self.feature_add('{}_mean'.format(prop), eledf.loc[:, prop].mean())
            self.feature_add('{}_mad'.format(prop), eledf.loc[:, prop].mad())

    def feature_add_weighted_average(self):
        # Load Elements.csv as DataFrame
        eledf = pd.read_csv(r'./Data/Elements_01Feb19.csv', index_col=1)

        # Slice Elements.csv based on elements present
        eledf = eledf.loc[list(self.elements_wt.keys())]

        def calc_weighted_average(a, b):
            num = np.sum(a * b)
            den = np.sum(b)
            return num/den

        for feature_name, feature_values in eledf.T.iterrows():
            feat = calc_weighted_average(feature_values.values, np.fromiter(self.elements_wt.values(), dtype=float))
            self.feature_add('{nm}_wtavg'.format(nm=feature_name), feat)

    def feature_add_unsupervised_properties(self):
        # Load Elements.csv as DataFrame, Slice Elements.csv based on elements present
        eledf = pd.read_csv(r'../Data/Elements_01Feb19.csv', index_col=1)
        eledf = eledf.loc[list(self.elements_wt.keys())]

        for feature_name, feature_values in eledf.T.iterrows():
            # Drop elements with 0 weight loading
            weights = np.fromiter(self.elements_wt.values(), dtype=float)
            feature_values = feature_values[weights != 0]
            weights = weights[weights != 0]

            # Calculate all values
            fwmean = np.sum(feature_values * weights) / np.sum(weights)
            avgdev = np.sum(weights * np.abs(feature_values - np.mean(feature_values))) / np.sum(weights)
            mx = np.max(feature_values)
            mn = np.min(feature_values)

            # Add all values to feature list
            self.feature_add('{}_mean'.format(feature_name), fwmean)
            self.feature_add('{}_mad'.format(feature_name), avgdev)
            self.feature_add('{}_max'.format(feature_name), mx)
            self.feature_add('{}_min'.format(feature_name), mn)
            self.feature_add('{}_rng'.format(feature_name), mx-mn)

    def feature_add_elemental_properties(self, mol_fraction=False):
        # Load Elements.csv as DataFrame, Slice Elements.csv based on elements present
        eledf = pd.read_csv(r'../Data/Elements_01Feb19.csv', index_col=1)
        eledf = eledf.loc[list(self.elements_wt.keys())]

        # Methods of processing different features
        def calc_valence(self, values, weights, feature_name):
            pass

        def calc_statistics(self, values, weights, feature_name):
            fwmean = np.sum(values * weights)/np.sum(weights)
            avgdev = np.sum(weights * np.abs(values - np.mean(values)))/np.sum(weights)  # (v25)
            # avgdev = np.sum(weights * np.abs(values - fwmean)) / np.sum(weights)  # (v51)
            self.feature_add('{}_mean'.format(feature_name), fwmean)
            self.feature_add('{}_mad'.format(feature_name), avgdev)

        def calc_zunger(self, values, weights, feature_name):
            # Attempt to set Ru value, flag -1 if no Ru
            try:
                Ru_Zpp = values['Ru']
                values.drop(index=['Ru'], inplace=True)
            except KeyError:
                Ru_Zpp = -1

            # Attempt to set K value, flag -1 if no K
            try:
                K_Zpp = values['K']
                values.drop(index=['K'], inplace=True)
            except KeyError:
                K_Zpp = -1

            # Try to set SE value, flag -1 if no SE
            try:
                SE_Zpp = values.values[0]
            except IndexError:
                SE_Zpp = -1

            # if no Ru, just ignore the sample (TODO: this is temporary for current project, but dumb in general)
            if Ru_Zpp == -1:
                return

            # if both flags are thrown, Zpp is futile
            if (SE_Zpp == -1) & (K_Zpp == -1):
                return

            if feature_name == 'Zunger Pseudopotential (pi)':
                if SE_Zpp == -1:
                    self.feature_add('Zunger Pseudopotential (pi)_Ru+K', Ru_Zpp + K_Zpp)
                elif K_Zpp == -1:
                    self.feature_add('Zunger Pseudopotential (pi)_Ru+SE', Ru_Zpp + SE_Zpp)
                else:
                    self.feature_add('Zunger Pseudopotential (pi)_Ru+SE', Ru_Zpp+SE_Zpp)
                    self.feature_add('Zunger Pseudopotential (pi)_Ru+K', Ru_Zpp + K_Zpp)
                    self.feature_add('Zunger Pseudopotential (pi)_SE+K', SE_Zpp+K_Zpp)

            elif feature_name == 'Zunger Pseudopotential (sigma)':
                if SE_Zpp == -1:
                    self.feature_add('Zunger Pseudopotential (sigma)_Ru-K', np.abs(Ru_Zpp - K_Zpp))
                elif K_Zpp == -1:
                    self.feature_add('Zunger Pseudopotential (sigma)_Ru-SE', np.abs(Ru_Zpp - SE_Zpp))
                else:
                    self.feature_add('Zunger Pseudopotential (sigma)_Ru-SE', np.abs(Ru_Zpp-SE_Zpp))
                    self.feature_add('Zunger Pseudopotential (sigma)_Ru-K', np.abs(Ru_Zpp-K_Zpp))
                    self.feature_add('Zunger Pseudopotential (sigma)_SE-K', np.abs(SE_Zpp-K_Zpp))

            else:
                print('Error adding Zpp')

        # Create Dictionary to process each feature differently
        process_dict = {
            'Atomic Number': calc_statistics,
            'Atomic Volume': calc_statistics,
            'Atomic Weight': calc_statistics,
            'Boiling Temperature': calc_statistics,
            'Periodic Table Column': calc_statistics,
            'Conductivity': calc_statistics,
            'Covalent Radius': calc_statistics,
            'Density': calc_statistics,
            'Dipole Polarizability': calc_statistics,
            'Electron Affinity': calc_statistics,
            'Electronegativity': calc_statistics,
            'Fusion Enthalpy': calc_statistics,
            'GS Bandgap': calc_statistics,
            'GS Energy': calc_statistics,
            'Heat Capacity (Mass)': calc_statistics,
            'Heat Capacity (Molar)': calc_statistics,
            'Heat Fusion': calc_statistics,
            'First Ionization Energy': calc_statistics,
            'Second Ionization Energy': calc_statistics,
            'Third Ionization Energy': calc_statistics,
            'Fourth Ionization Energy': calc_statistics,
            'Fifth Ionization Energy': calc_statistics,
            'Sixth Ionization Energy': calc_statistics,
            'Seventh Ionization Energy': calc_statistics,
            'Eighth Ionization Energy': calc_statistics,
            'IsAlkali': calc_statistics,
            'IsDBlock': calc_statistics,
            'IsFBlock': calc_statistics,
            'IsMetal': calc_statistics,
            'IsMetalloid': calc_statistics,
            'IsNonmetal': calc_statistics,
            'Melting Temperature': calc_statistics,
            'Mendeleev Number': calc_statistics,
            'Number d-shell Unfilled Electrons': calc_statistics,
            'Number d-shell Valence Electrons': calc_statistics,
            'Number f-shell Unfilled Electrons': calc_statistics,
            'Number f-shell Valence Electrons': calc_statistics,
            'Number p-shell Unfilled Electrons': calc_statistics,
            'Number p-shell Valence Electrons': calc_statistics,
            'Number s-shell Unfilled Electrons': calc_statistics,
            'Number s-shell Valence Electrons': calc_statistics,
            'Number Unfilled Electrons': calc_statistics,
            'Number Valence Electrons': calc_statistics,
            'Polarizability': calc_statistics,
            'Periodic Table Row': calc_statistics,
            'Phi': calc_statistics,
            'Zunger Pseudopotential (d)': calc_statistics,
            'Zunger Pseudopotential (p)': calc_statistics,
            'Zunger Pseudopotential (pi)': calc_zunger,
            'Zunger Pseudopotential (s)': calc_statistics,
            'Zunger Pseudopotential (sigma)': calc_zunger,
        }

        if mol_fraction:
            for feature_name, feature_values in eledf.T.iterrows():
                process_dict.get(feature_name,
                                 lambda a,b,c,d: print('Feature Name ({}) Not Found'.format(feature_name)))(
                    self,
                    feature_values,
                    np.fromiter(self.elements_mol.values(), dtype=float),
                    feature_name
                )
        else:
            for feature_name, feature_values in eledf.T.iterrows():
                process_dict.get(feature_name,
                                 lambda a,b,c,d: print('Feature Name ({}) Not Found'.format(feature_name)))(
                    self,
                    feature_values,
                    np.fromiter(self.elements_wt.values(), dtype=float),
                    feature_name
                )

    def feature_add_Lp_norms(self):
        vals = np.fromiter(self.elements_wt.values(), dtype=float)
        sm = np.sum(vals)

        for p in [2, 3, 5, 7, 10]:
            pnorm = np.sum(np.power(vals / sm, p))**(1/p)
            self.feature_add(key='Lp{}'.format(p), value=pnorm)

    def feature_add_n_elements(self):
        n_eles = 0
        for val in self.elements_wt.values():
            if val > 0:
                n_eles += 1

        self.feature_add('n_elements',n_eles)

    def calc_mole_fraction(self):
        eledf = pd.read_csv(r'../Data/Elements_01Feb19.csv', index_col=1)

        alumina_mol = (1 - np.sum(list(self.elements_wt.values())) / 100) * 2 / 101.96 # mol of alumina

        for ele, wt in self.elements_wt.items():
            at_wt = eledf.loc[eledf.index == ele, 'Atomic Weight'].values[0]
            self.elements_mol[ele] = wt / 100 * 2 / at_wt

        tot_mol = np.sum(list(self.elements_mol.values()) + [alumina_mol])

        for ele, wt in self.elements_mol.items():
            self.elements_mol[ele] = np.round(wt / tot_mol * 100, 2)

class CatalystObservation():
    def __init__(self):
        self.temperature = None
        self.pressure = None
        self.space_velocity = None
        self.gas = None
        self.concentration = None
        self.reactor = None

        self.activity = None
        self.selectivity = None
        self.activity_error = None
        self.selectivity_error = None

    def to_dict(self):
        dict = {
            'temperature': self.temperature,
            'pressure': self.pressure,
            'space_velocity': self.space_velocity,
            'gas': self.gas,
            'ammonia_concentration': self.concentration,
            'reactor': self.reactor,
            'Measured Conversion': self.activity, # TODO lower case and fix propigation through code
            'measured conversion experimental error': self.activity_error,
            'selectivity': self.selectivity,
            'selectivity experimental error': self.selectivity_error
        }

        return dict
