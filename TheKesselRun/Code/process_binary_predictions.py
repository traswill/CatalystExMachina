import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def raw_to_processed(pth):
    dat_df = pd.read_csv('{}.csv'.format(pth), index_col=0)
    nm_df = dat_df.loc[:, dat_df.columns.str.contains('Loading')]
    df = pd.DataFrame(dat_df['Predictions'])

    for inx, vals in nm_df.iterrows():
        vals = vals[vals != 0]
        if len(vals) == 2:
            continue
        vals.index = [item[0] for item in vals.index.str.split(' ').values]
        df.loc[inx, 'Element 1'] = vals.index[0]
        df.loc[inx, 'Loading 1'] = vals[0]
        df.loc[inx, 'Element 2'] = vals.index[1]
        df.loc[inx, 'Loading 2'] = vals[1]
        df.loc[inx, 'Element 3'] = vals.index[2]
        df.loc[inx, 'Loading 3'] = vals[2]
        df.loc[inx, 'Name'] = '{}{} {}{} {}{}'.format(vals.index[0], vals[0], vals.index[1], vals[1], vals.index[2], vals[2])

    df.to_csv('{}-Processed.csv'.format(pth))

def plot_heatmap():
    pass
    # df = pd.read_csv('.//BP-v13-Processed.csv', index_col=0)
    # sbdf = df[['Predictions']].dropna()
    # print(sbdf)
    #
    #
    # sns.set(style="white")
    # sns.heatmap(sbdf)
    # plt.show()
    # exit()
    #
    # # Compute the correlation matrix
    # corr = df.corr()
    #
    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    #
    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))
    #
    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #
    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})

if __name__ == '__main__':
    # raw_to_processed(pth='../Results/v15-pred_r/v15-pred-BinaryPredictions-300C')
    raw_to_processed(pth='../Results/v15-pred_r/v15-pred-HalfRu-300C')

