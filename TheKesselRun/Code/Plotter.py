import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, BoundaryNorm, to_hex, Normalize
from matplotlib.cm import get_cmap

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker, CustomJS, Slider, Select
from bokeh.plotting import figure, show, output_file, save, curdoc
import bokeh.palettes as pals
from bokeh.models import Range1d, DataRange1d
from bokeh.layouts import row, widgetbox, column, layout

class Graphic():
    def __init__(self, learner):
        self.learner = learner
        self.graphdf = learner.plot_df

        if self.graphdf.empty:
            print('Graphic Failed to initialize.  Learner does not contain valid plot dataframe.')

        # Full descriptive name X(#)Y(#)Z(#)
        self.graphdf['Name'] = [
            ''.join('{}({})'.format(key, str(int(val)))
                    for key, val in x) for x in self.graphdf['Element Dictionary']
        ]

        for index, edict in self.graphdf['Element Dictionary'].iteritems():
            self.graphdf.loc[index, 'Name'] = ''.join('{}({})'.format(key, str(int(val))) for key, val in edict)

            i = 1
            for key, val in edict:
                self.graphdf.loc[index, 'Ele{}'.format(i)] = key
                self.graphdf.loc[index, 'Load{}'.format(i)] = val
                i += 1

        # Catalyst ID
        self.graphdf['ID'] = [int(nm.split('_')[0]) for nm in self.graphdf.index.values]

        # Remove Dictionary to avoid problems down the line
        self.graphdf.drop(columns='Element Dictionary', inplace=True)

        self.graphdf['clr'] = 'b'

        # # Create hues for heatmaps
        # def create_feature_hues(self, feature):
        #     try:
        #         unique_feature = np.unique(self.slave_dataset.loc[:, feature].values)
        #     except KeyError:
        #         print('KeyError: {} not found'.format(feature))
        #         return
        #
        #     n_feature = len(unique_feature)
        #     max_feature = np.max(unique_feature)
        #     min_feature = np.min(unique_feature)
        #
        #     if max_feature == min_feature:
        #         self.plot_df['{}_hues'.format(feature)] = "#3498db"  # Blue!
        #     else:
        #         palette = sns.color_palette('plasma', n_colors=n_feature+1)
        #         self.plot_df['{}_hues'.format(feature)] = [
        #             palette[i] for i in [int(n_feature * (float(x) - min_feature) / (max_feature - min_feature))
        #                                       for x in self.slave_dataset.loc[:, feature].values]
        #         ]
        #
        # self.plot_df['temperature_hues'] = 0
        #
        # # Grab top 10 features, add hues to plotdf
        # try:
        #     feature_rank = self.extract_important_features()
        #     for feat in feature_rank.sort_values(by='Feature Importance', ascending=False).head(10).index.values:
        #         create_feature_hues(self, feat)
        # except AttributeError:
        #     print('Learner does not support feature extraction.')
        #
        # # Process Second Element Colors
        # uniq_eles = np.unique(self.plot_df['Ele2'])
        # n_uniq = len(uniq_eles)
        # palette = sns.color_palette('tab20', n_colors=n_uniq + 1)
        # self.plot_df['Ele2_hues'] = [
        #     palette[np.where(uniq_eles == i)[0][0]] for i in self.plot_df['Ele2']
        # ]
        #
        # return self.plot_df

    def set_color(self, feature, min=None, max=None):
        vals = self.graphdf[feature]

        cmap = get_cmap('plasma')
        if (min is None) and (max is None):
            norm = Normalize(vmin=np.min(vals), vmax=np.max(vals))
        else:
            norm = Normalize(vmin=min, vmax=max)

        self.graphdf['clr'] = cmap(norm(self.graphdf[feature].values))

    def plot_basic(self):
        uniq_tmps = np.unique(self.graphdf['temperature'])

        for tmp in uniq_tmps:
            plt.scatter(x=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'Predicted Conversion'],
                        y=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'Measured Conversion'],
                        c=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'clr'],
                        label='{}{}C'.format(int(tmp), u'\N{DEGREE SIGN}'),
                        edgecolors='k')

        plt.xlabel('Predicted Conversion')
        plt.ylabel('Measured Conversion')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(title='Temperature')
        plt.tight_layout()

        if len(uniq_tmps) == 1:
            plt.savefig('{}//figures//{}-basic-{}C.png'.format(self.learner.svfl, self.learner.svnm, uniq_temps[0]),
                        dpi=400)
        else:
            plt.savefig('{}//figures//{}-basic.png'.format(self.learner.svfl, self.learner.svnm),
                        dpi=400)
        plt.close()

    def plot_metadata(self):
        pass

    def plot_err(self):
        df = pd.DataFrame([self.predictions,
                           self.labels_df.values,
                           self.plot_df['{}_hues'.format('temperature')].values,
                           self.plot_df['{}'.format('temperature')].values],
                          index=['pred', 'meas', 'clr', 'feat']).T

        rats = np.abs(np.subtract(self.predictions, self.labels_df.values, out=np.zeros_like(self.predictions),
                                  where=self.labels_df.values != 0))

        rat_count = rats.size
        wi5 = (rats < 0.05).sum()
        wi10 = (rats < 0.10).sum()
        wi20 = (rats < 0.20).sum()

        fig, ax = plt.subplots()

        uniq_features = np.unique(df['feat'])

        # Katie's Colors
        # color_selector = {
        #     250: 'purple',
        #     300: 'darkgreen',
        #     350: 'xkcd:coral',
        #     400: 'darkblue',
        #     450: 'xkcd:salmon'
        # }

        cmap = get_cmap('plasma')
        norm = Normalize(vmin=250, vmax=450)

        color_selector = {
            250: cmap(norm(250)),
            300: cmap(norm(300)),
            350: cmap(norm(350)),
            400: cmap(norm(400)),
            450: cmap(norm(450))
        }

        if len(uniq_features) == 1:

            ax.scatter(x=df.loc[df['feat'] == uniq_features[0], 'pred'],
                       y=df.loc[df['feat'] == uniq_features[0], 'meas'],
                       c=color_selector.get(uniq_features[0]),
                       label='{}{}C'.format(int(uniq_features[0]), u'\N{DEGREE SIGN}'),
                       edgecolors='k')
        else:
            for feat in uniq_features:
                ax.scatter(x=df.loc[df['feat'] == feat, 'pred'],
                           y=df.loc[df['feat'] == feat, 'meas'],
                           c=color_selector.get(feat),  # df.loc[df['feat'] == feat, 'clr'],
                           label='{}{}C'.format(int(feat), u'\N{DEGREE SIGN}'),
                           edgecolors='k')

        x = np.array([0, 0.5, 1])
        y = np.array([0, 0.5, 1])

        ax.plot(x, y, lw=2, c='k')
        ax.fill_between(x, y + 0.1, y - 0.1, alpha=0.1, color='b')
        ax.fill_between(x, y + 0.2, y + 0.1, alpha=0.1, color='y')
        ax.fill_between(x, y - 0.2, y - 0.1, alpha=0.1, color='y')

        if metadata:
            plt.figtext(0.99, 0.01,
                        'Within 5%: {five:0.2f} \nWithin 10%: {ten:0.2f} \nWithin 20%: {twenty:0.2f}'.format(
                            five=wi5 / rat_count, ten=wi10 / rat_count, twenty=wi20 / rat_count),
                        horizontalalignment='right', fontsize=6)

            mean_abs_err = mean_absolute_error(self.labels_df.values, self.predictions)
            rmse = np.sqrt(mean_squared_error(self.labels_df.values, self.predictions))

            plt.figtext(0, 0.01, 'MeanAbsErr: {:0.2f} \nRMSE: {:0.2f}'.format(mean_abs_err, rmse),
                        horizontalalignment='left', fontsize=6)

            plt.figtext(0.5, 0.01, 'E{} A{} S{}'.format(self.element_filter, self.ammonia_filter, self.sv_filter),
                        horizontalalignment='center', fontsize=6)

        plt.xlabel('Predicted Conversion')
        plt.ylabel('Measured Conversion')
        plt.legend(title='Temperature')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plt.title('{}-{}'.format(self.svnm, 'err'))
        plt.legend(title='Temperature')
        plt.tight_layout()
        plt.savefig('{}//figures//{}-{}.png'.format(self.svfl, self.svnm, 'err'),
                    dpi=400)
        plt.close()


