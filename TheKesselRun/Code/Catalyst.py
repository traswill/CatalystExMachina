# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np


class CatalystObject():
    """CatalystObject will contain each individual training set"""
    def __init__(self):
        self.ID = None
        self.activity = None
        self.group = None

        self.observation_dict = dict()
        self.input_dict = dict() # TODO: deprecated, remove
        self.feature_dict = dict()
        self.elements = dict()

    def add_observation(self, temperature=None, space_velocity=None, gas=None, gas_concentration=None, pressure=None,
                        reactor_number=None, activity=None, selectivity=None):
        obs = CatalystObservation()
        obs.temperature = temperature
        obs.space_velocity = space_velocity
        obs.gas = gas
        obs.concentration = gas_concentration
        obs.pressure = pressure
        obs.reactor = reactor_number
        obs.activity = activity
        obs.selectivity = selectivity

        self.observation_dict[len(self.observation_dict)] = obs

    def input_temperature(self, T): # TODO: deprecated, remove
        self.input_dict['temperature'] = T

    def add_element(self, element, weight_loading):
        if (element != '-') & (element != '--'):
            self.elements[element] = weight_loading

    def input_space_velocity(self, space_velocity): # TODO: deprecated, remove
        self.input_dict['space_velocity'] = space_velocity

    def input_reactor_number(self, reactor_number): # TODO: deprecated, remove
        self.input_dict['reactor_number'] = reactor_number

    def input_ammonia_concentration(self, ammonia_concentration): # TODO: deprecated, remove
        self.input_dict['ammonia_concentration'] = ammonia_concentration

    def input_standard_error(self, error): # TODO: deprecated, remove
        self.input_dict['standard error'] = error

    def input_n_averaged_samples(self, n_avg): # TODO: deprecated, remove
        self.input_dict['n_averaged'] = n_avg

    def input_n_Cl_atoms(self, Cl_atoms):
        self.input_dict['n_Cl_atoms'] = Cl_atoms

    def input_group(self, group): # TODO: change this to reflect updated class structure (self.group)
        self.input_dict['group'] = group

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

    def feature_add_unsupervised_properties(self):
        # Load Elements.csv as DataFrame, Slice Elements.csv based on elements present
        eledf = pd.read_csv(r'../Data/Elements_Unsupervised.csv', index_col=1)
        eledf = eledf.loc[list(self.elements.keys())]

        for feature_name, feature_values in eledf.T.iterrows():
            # Drop elements with 0 weight loading
            weights = np.fromiter(self.elements.values(), dtype=float)
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

    def feature_add_elemental_properties(self):
        # Load Elements.csv as DataFrame, Slice Elements.csv based on elements present
        eledf = pd.read_csv(r'../Data/Elements_Cleaned.csv', index_col=1)
        eledf = eledf.loc[list(self.elements.keys())]

        # Methods of processing different features
        def calc_valence(self, values, weights, feature_name):
            pass

        def calc_statistics(self, values, weights, feature_name):
            # self.feature_add('{}_mean'.format(feature_name), np.mean(values))
            # self.feature_add('{}_mad'.format(feature_name), np.mean(np.abs(values-np.mean(values))))
            fwmean = np.sum(values * weights)/np.sum(weights)
            avgdev = np.sum(weights * np.abs(values - np.mean(values)))/np.sum(weights) #(v25)
            # avgdev = np.sum(np.abs(weights * values - fwmean)) / np.sum(weights)
            self.feature_add('{}_wt-mean'.format(feature_name), fwmean)
            self.feature_add('{}_wt-mad'.format(feature_name), avgdev)
            # self.feature_add('{}_min'.format(feature_name), np.max(values))
            # self.feature_add('{}_max'.format(feature_name), np.min(values))
            # self.feature_add('{}_rng'.format(feature_name), np.max(values)-np.min(values))

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

    def feature_add_Lp_norms(self):
        vals = np.fromiter(self.elements.values(), dtype=float)
        sm = np.sum(vals)

        for p in [2, 3, 5, 7, 10]:
            pnorm = np.sum(np.power(vals / sm, p))**(1/p)
            self.feature_add(key='Lp{}'.format(p), value=pnorm)

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

    def feature_add_xrd_peaks(self, peak_xx, peak_yy):
        # Adds peak intensities
        peak_xx = ['XRD 2TH {:0.2f}'.format(x) for x in peak_xx]
        bundle_data = list(zip(peak_xx, peak_yy))

        for x, y in bundle_data:
            self.feature_add(key=x, value=y)

    def feature_add_xrd_peak_FWHM(self, peak_nm, peak_fwhm):
        self.feature_add(key=peak_nm, value=peak_fwhm)


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