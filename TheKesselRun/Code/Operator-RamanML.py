from TheKesselRun.Code.LearnerOrder import SupervisedLearner, CatalystContainer
from TheKesselRun.Code.LearnerAnarchy import Anarchy
from TheKesselRun.Code.Catalyst import CatalystObject, CatalystObservation
from TheKesselRun.Code.Plotter import Graphic
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from sklearn.metrics import r2_score, explained_variance_score, \
        mean_absolute_error, roc_curve, recall_score, precision_score, mean_squared_error, accuracy_score

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import time
import scipy.cluster
import glob
import random
import os
import peakutils

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, GroupKFold, LeaveOneGroupOut, \
    LeaveOneOut, learning_curve


def load_nh3_catalysts(catcont, drop_empty_columns=True):
    """ Import NH3 data from Katie's HiTp dataset(cleaned). """
    df = pd.read_csv(r"..\Data\Processed\AllData_Condensed.csv", index_col=0)
    df.dropna(axis=0, inplace=True, how='all')

    # Loop through all data
    for index, dat in df.iterrows():
        # If the ID already exists in container, then only add an observation.  Else, generate a new catalyst.
        if dat['ID'] in catcont.catalyst_dictionary:
            catcont.catalyst_dictionary[dat['ID']].add_observation(
                temperature=dat['Temperature'],
                space_velocity=dat['Space Velocity'],
                gas=None,
                gas_concentration=dat['NH3'],
                pressure=None,
                reactor_number=int(dat['Reactor']),
                activity=dat['Conversion'],
                selectivity=None
            )
        else:
            cat = CatalystObject()

            # Set up elements
            cat.ID = dat['ID']
            cat.add_element(dat['Ele1'], dat['Wt1'])
            cat.add_element(dat['Ele2'], dat['Wt2'])
            cat.add_element(dat['Ele3'], dat['Wt3'])
            cat.calc_mole_fraction()
            cat.set_group(dat['Groups'])
            cat.feature_add_n_elements()
            cat.feature_add_Lp_norms()
            cat.feature_add_elemental_properties(mol_fraction=True)

            # Input Raman Data
            raman_root = r'C:\Users\quick\OneDrive - University of South Carolina\Data\Proc - Raman Data\Raman ML Project\Data'
            pths = glob.glob('{}\{}_*.csv'.format(raman_root, dat['ID']))

            if pths:
                base_pth = '\\'.join(pths[0].split('\\')[:-1])

                raman_df = pd.read_csv('{}\\{}_638nm.csv'.format(base_pth, dat['ID']), index_col=0)
                cat.add_638nm_raman(raman_df.index.values, np.array([x[0] for x in raman_df.values]))

                raman_df = pd.read_csv('{}\\{}_473nm.csv'.format(base_pth, dat['ID']), index_col=0)
                cat.add_473nm_raman(raman_df.index.values, np.array([x[0] for x in raman_df.values]))

            # Input XRD Data
            xrd_root = r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Data\XRD Data Cleaned'
            pth = glob.glob('{}\{} *.txt'.format(xrd_root, dat['ID']))

            if pth:
                xrd_df = pd.read_csv(pth[0], index_col=0, header=None)
                cat.add_xrd(xrd_df.index.values, np.array([x[0] for x in xrd_df.values]))

            cat.add_observation(
                temperature=dat['Temperature'],
                space_velocity=dat['Space Velocity'],
                gas=None,
                gas_concentration=dat['NH3'],
                pressure=None,
                reactor_number=int(dat['Reactor']),
                activity=dat['Conversion'],
                selectivity=None
            )

            catcont.add_catalyst(index=cat.ID, catalyst=cat)

def reduce_spectra(catcont, spectype=None, n_comp=2):
    spec = pd.DataFrame()

    for catid, cat in catcont.catalyst_dictionary.items():
        if spectype == '638nm':
            temp_df = pd.DataFrame(data=cat.spectrum638_y, index=cat.spectrum638_x, columns=[catid])
        elif spectype == '473nm':
            temp_df = pd.DataFrame(data=cat.spectrum473_y, index=cat.spectrum473_x, columns=[catid])
        elif spectype == 'xrd':
            temp_df = pd.DataFrame(data=cat.spectrumXRD_y, index=cat.spectrumXRD_x, columns=[catid])
        else:
            print('No spectype specified. Exiting...')
            return

        spec = pd.concat([spec, temp_df], axis=1)

    spec.dropna(inplace=True, axis=1, how='all')  # Drop catalysts without spectra
    spec.fillna(value=-1, inplace=True)           # Fill -1 flag for NaNs in XRD pattern... TODO handle this better
    X = spec.values

    pca = PCA(n_components=n_comp)
    pca.fit(X)
    specpca = pd.DataFrame(list(zip(*pca.components_)), index=spec.T.index)

    for idx, vals in specpca.iterrows():
        for ii, val in enumerate(vals):
            catcont.catalyst_dictionary[idx].spectral_add('{}_PCA{}'.format(spectype, ii), val)

def bsub_and_add_spec(catcont, spectype=None, n_comp=2):
    spec = pd.DataFrame()

    for catid, cat in catcont.catalyst_dictionary.items():
        if spectype == '638nm':
            temp_df = pd.DataFrame(data=cat.spectrum638_y, index=cat.spectrum638_x, columns=[catid])
        elif spectype == '473nm':
            temp_df = pd.DataFrame(data=cat.spectrum473_y, index=cat.spectrum473_x, columns=[catid])
        elif spectype == 'xrd':
            temp_df = pd.DataFrame(data=cat.spectrumXRD_y, index=cat.spectrumXRD_x, columns=[catid])
        else:
            print('No spectype specified. Exiting...')
            return

        if temp_df.empty:
            continue

        x = temp_df.index.values
        y = temp_df.values

        y_background = peakutils.baseline(y, deg=3)

        fig, ax = plt.subplots(ncols=2)

        ax[0].plot(x, y)
        ax[0].plot(x, y_background, '--r')
        ax[1].plot(x, y - y_background)

        plt.suptitle(catid)
        rootpth = r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v97 - Spectral Bsub Results'
        plt.savefig('{}\\{}_{}.png'.format(rootpth, catid, spectype))
        plt.close()

        spec = pd.concat([spec, temp_df], axis=1)

    spec.dropna(inplace=True, axis=1, how='all')  # Drop catalysts without spectra
    spec.fillna(value=-1, inplace=True)  # Fill -1 flag for NaNs in XRD pattern... TODO handle this better
    X = spec.values

    pca = PCA(n_components=n_comp)
    pca.fit(X)
    specpca = pd.DataFrame(list(zip(*pca.components_)), index=spec.T.index)

    for idx, vals in specpca.iterrows():
        for ii, val in enumerate(vals):
            catcont.catalyst_dictionary[idx].spectral_add('{}_PCA{}'.format(spectype, ii), val)


if __name__ == '__main__':
    version = 'v97 - Spectral Spelunking N=5'
    note = 'Added Spectroscopy ML Code'

    catcont = CatalystContainer()
    load_nh3_catalysts(catcont)

    N = 5
    bsub_and_add_spec(catcont, spectype='638nm', n_comp=N)
    bsub_and_add_spec(catcont, spectype='473nm', n_comp=N)
    bsub_and_add_spec(catcont, spectype='xrd', n_comp=N)

    exit()

    catcont.build_master_container()

    # Init Learner
    skynet = SupervisedLearner(version=version, note=note)
    skynet.set_filters(
        element_filter=3,
        temperature_filter='350orless',
        ammonia_filter=1,
        space_vel_filter=2000,
        ru_filter=0,
        pressure_filter=None
    )

    # Set algorithm and add data
    skynet.set_learner(learner='etr', params='etr')
    skynet.load_static_dataset(catalyst_container=catcont)
    skynet.static_dataset = skynet.static_dataset[skynet.static_dataset['K Loading'] == 0.12]
    skynet.static_dataset = skynet.static_dataset[skynet.static_dataset['473nm_PCA0'] != 0]

    # # Set parameters
    targ_cols = list(['473nm_PCA{}'.format(i) for i in range(N)]) + \
                list(['638nm_PCA{}'.format(i) for i in range(N)]) + \
                list(['xrd_PCA{}'.format(i) for i in range(N)])

    skynet.set_target_columns(cols=targ_cols)
    skynet.set_group_columns(cols=['group'])
    skynet.set_hold_columns(cols=['Element Dictionary', 'ID', 'Measured Conversion'])

    skynet.filter_static_dataset()

    skynet.predict_crossvalidate(kfold=3)
    skynet.evaluate_regression_learner()

    skynet.result_dataset = skynet.dynamic_dataset[skynet.features_df.columns].copy()

    skynet.result_dataset = pd.concat(
        [skynet.result_dataset,
         pd.DataFrame(skynet.labels, index=skynet.labels_df.index, columns=skynet.target_columns)], axis=1)

    skynet.result_dataset = pd.concat(
        [skynet.result_dataset,
         pd.DataFrame(skynet.predictions, index=skynet.labels_df.index,
                      columns=['{}_predicted'.format(x) for x in skynet.target_columns])], axis=1)

    skynet.result_dataset['Measured Conversion'] = skynet.hold_df['Measured Conversion']

    skynet.result_dataset.to_csv('{}\\result_dataset-{}.csv'.format(skynet.svfl, skynet.svnm))
