import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import peakutils
from scipy.interpolate import UnivariateSpline


def load_data():
    # Process 052118
    df = pd.read_csv(r'../Data/RAW/052118-WAXS/Run List.csv')

    for idx, rw in df.iterrows():
        nm = rw['Sample Description']
        regex = re.search('\d+', nm)

        if regex is None:
            continue
        elif regex.end() < 5:
            df.loc[idx, 'ID'] = regex.group()

    nmdf = df[['Filename', 'ID']]

    # Process 052218
    df = pd.read_csv(r'../Data/RAW/052218-WAXS/Katie-WAXS.csv')

    for idx, rw in df.iterrows():
        nm = rw['Sample Description']
        regex = re.search('\d+', nm)
        if regex is None:
            continue
        elif regex.end() < 5:
            df.loc[idx, 'ID'] = regex.group()

    nmdf = pd.concat([nmdf, df[['Filename', 'ID']]])
    nmdf.dropna(inplace=True)
    return nmdf

def subtract_background(df):
    outdf = pd.DataFrame()
    for rw in df.iteritems():
        id = rw[0].replace(' Intensity', '')
        dat = rw[1]
        xx = dat.index
        yy = dat.values

        bline = peakutils.baseline(y=yy, deg=2)
        tdf = pd.DataFrame(yy - bline, index=xx, columns=[id])
        outdf = pd.concat([outdf, tdf], axis=1)
    return outdf

def extract_peak_indx(df):
    xx = df.index
    yy = df.sum(axis=1).values
    indx = peakutils.indexes(yy, thres=0.05, min_dist=4)
    df.iloc[indx].to_csv(r'../Data/Processed/WAXS/WAXS_Peak_Extraction.csv')


def extract_auc(df):
    # TODO
    def auc_from_bounds(data, low, high):
        wrkdf = data[(data.index > low) & (data.index < high)]
        auc = np.trapz(wrkdf.values, wrkdf.index, dx=0.1)
        return auc


def extract_fwhm(df):
    # Hardcoded
    lst = [
        (15.5, 16.75, '16 2TH'), (17, 18.5, '18 2TH'), (21.5, 22.5, '22 2TH'), (25, 27.5, '26 2TH'),
        (27.5, 29.75, '28 2TH'), (34.75, 37, '36 2TH'), (40.25, 42.75, '41 2TH'), (44, 48, '46 2TH'),
        (50, 51.25, '51 2TH')
    ]

    outdf = pd.DataFrame()

    for nm, ds in df.iteritems():
        xx = ds.index
        yy = ds.values

        for low, high, twoth in lst:
            mask = (xx > low) & (xx < high)
            x = xx[mask]
            y = yy[mask] - peakutils.baseline(y=yy[mask], deg=1)
            spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
            try:
                r1, r2 = spline.roots()
            except ValueError:
                r1 = 0
                r2 = 0

            fwhm = r2 - r1
            outdf.loc[nm, twoth] = fwhm

    outdf.to_csv(r'../Data/Processed/WAXS/WAXS_FWHM_Extraction.csv')

if __name__ == '__main__':
    nmdf = load_data()
    df = pd.DataFrame()

    for idx, rw in nmdf.iterrows():
        nm = rw['Filename'].strip()
        id = rw['ID']
        try:
            pth = glob.glob('..\Data\RAW\*\{}.txt'.format(nm))[0]
        except IndexError:
            print('Failed to load: {}'.format(nm))
            continue

        datdf = pd.read_csv(pth, names=['{} Intensity'.format(id)], header=0, index_col=0)
        datdf.to_csv(r'../Data/Processed/WAXS/{}_waxs.csv'.format(id))

        df = pd.concat([df, datdf], axis=1)

    # Zero-baseline
    idf = df[(df.index > 12) & (df.index < 24)]
    for rw in idf.iteritems():
        df[rw[0]] -= rw[1].min()

    # Normalize by 40-42 twotheta peak
    idf = df[(df.index > 12) & (df.index < 24)]
    for rw in idf.iteritems():
        idx = rw[0]
        vals = rw[1]
        auc = np.trapz(vals.values, vals.index, dx=0.1)
        df[idx] = df[idx]/auc

    # Extract auc values for ML
    df = subtract_background(df)
    extract_peak_indx(df)
    extract_fwhm(df)

    df.plot(legend=False)
    plt.show()
