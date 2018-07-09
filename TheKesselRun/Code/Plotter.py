# Created by Travis Williams
# Property of the University of South Carolina
# Jochen Lauterbach Group
# Contact: travisw@email.sc.edu
# Project Start: February 15, 2018

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, BoundaryNorm, to_hex, Normalize
from matplotlib import  cm

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Whisker, CustomJS, Slider, Select
from bokeh.plotting import figure, show, output_file, save, curdoc
import bokeh.palettes as pals
from bokeh.models import Range1d, DataRange1d
from bokeh.layouts import row, widgetbox, column, layout


class Graphic():
    def __init__(self, learner, df=None):
        sns.set(palette='plasma', context='paper', style='white')
        self.learner = learner

        if df is None:
            self.graphdf = self.learner.slave_dataset.copy()
            self.graphdf['Predicted Conversion'] = self.learner.predictions

            if self.graphdf.empty:
                print('Graphic Failed to initialize.  Learner does not contain valid plot dataframe.')
            else:
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
        else:
            self.graphdf = df

    def set_color(self, feature, cbnds=None, cmap='plasma'):
        if cbnds is not None:
            mn, mx = cbnds
        else:
            mn, mx = (None, None)

        vals = self.graphdf[feature]

        if (mn is None) and (mx is None):
            norm = Normalize(vmin=np.min(vals), vmax=np.max(vals))
        else:
            norm = Normalize(vmin=mn, vmax=mx)

        c = cm.ScalarMappable(norm=norm, cmap=cmap).get_cmap()
        self.graphdf['clr'] = self.graphdf[feature].apply(norm).apply(c)

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
            plt.savefig('{}//figures//{}-basic-{}C.png'.format(self.learner.svfl, self.learner.svnm, uniq_tmps[0]),
                        dpi=400)
        else:
            plt.savefig('{}//figures//{}-basic.png'.format(self.learner.svfl, self.learner.svnm),
                        dpi=400)
        plt.close()

    def plot_metadata(self):
        pass

    def plot_err(self, metadata=True, svnm=None, color_bounds=None):
        fig, ax = plt.subplots()

        rats = np.abs(np.subtract(self.graphdf['Predicted Conversion'], self.graphdf['Measured Conversion'],
                                  out=np.zeros_like(self.graphdf['Predicted Conversion']),
                                  where=self.graphdf['Measured Conversion'] != 0))

        rat_count = rats.size
        wi5 = (rats < 0.05).sum()
        wi10 = (rats < 0.10).sum()
        wi20 = (rats < 0.20).sum()

        self.set_color(feature='temperature', cbnds=color_bounds)
        uniq_tmps = np.unique(self.graphdf['temperature'])

        for tmp in uniq_tmps:
            plt.scatter(x=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'Predicted Conversion'],
                        y=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'Measured Conversion'],
                        c=self.graphdf.loc[self.graphdf['temperature'] == tmp, 'clr'].values,
                        label='{}{}C'.format(int(tmp), u'\N{DEGREE SIGN}'),
                        edgecolors='k', linewidth=1)

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

            mean_abs_err = mean_absolute_error(self.graphdf['Measured Conversion'], self.graphdf['Predicted Conversion'])
            rmse = np.sqrt(mean_squared_error(self.graphdf['Measured Conversion'], self.graphdf['Predicted Conversion']))

            plt.figtext(0, 0.01, 'MeanAbsErr: {:0.2f} \nRMSE: {:0.2f}'.format(mean_abs_err, rmse),
                        horizontalalignment='left', fontsize=6)

            plt.figtext(0.5, 0.01, 'E{} A{} S{}'.format(self.learner.element_filter,
                                                        self.learner.ammonia_filter,
                                                        self.learner.sv_filter),
                        horizontalalignment='center', fontsize=6)

        plt.xlabel('Predicted Conversion')
        plt.ylabel('Measured Conversion')
        plt.legend(title='Temperature')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(title='Temperature')
        plt.tight_layout()
        if svnm is None:
            plt.savefig('{}//figures//{}-{}.png'.format(self.learner.svfl, self.learner.svnm, 'err'),
                        dpi=400)
        else:
            plt.savefig('{}//figures//{}.png'.format(self.learner.svfl, svnm), dpi=400)
        plt.close()

    def plot_kernel_density(self, feat_list=None, margins=True, element=None):
        """

        :param feat_list: A list of features to be plotted
        :param margins: Whether to have KDE 1D plots in the margins or plane plots
        :return:
        """

        if feat_list is None:
            self.learner.feature_importance_df.sort_values(by='Feature Importance', inplace=True, ascending=False)
            top_features = self.learner.feature_importance_df.head().index

        for feat in feat_list:
            if margins:
                g = sns.jointplot(x=feat, y='Measured Conversion',  data=self.learner.slave_dataset, kind='kde', color='k',
                                  stat_func=None)
                g.plot_joint(plt.scatter, c='w', s=15, linewidth=1, marker=".")
                g.ax_joint.collections[0].set_alpha(0)
            else:
                fig, ax = plt.subplots(figsize=(5,5))
                sns.kdeplot(self.learner.slave_dataset[feat], self.learner.slave_dataset['Measured Conversion'],
                                cmap='Greys', shade=True, shade_lowest=False, ax=ax)

                ax.scatter(self.learner.slave_dataset[feat], self.learner.slave_dataset['Measured Conversion'],
                            c='w', s=15, marker='.')

                if element is not None:
                    df = self.learner.slave_dataset[self.learner.slave_dataset['{} Loading'.format(element)] != 0].copy()
                    ax.scatter(df[feat], df['Measured Conversion'], c='r', s=15, marker='x')

            # Modifying things for publication
            lim_dict = {
                '': '',
                # 'Number d-shell Valance Electrons_wt-mad': plt.xlim(-2.5, 17.5),
                # 'Periodic Table Column_wt-mad': plt.xlim(0, 35),
                # 'Second Ionization Energy_wt-mad': plt.xlim(500, 700)
            }

            lim_dict.get(feat, plt.autoscale())
            plt.ylim(-0.3, 1.3)

            plt.xlabel(feat.split('_')[0])

            plt.tight_layout()
            if element is not None:
                plt.savefig('{}//Figures//{}-{}-{}'.format(self.learner.svfl, feat, self.learner.svnm, element), dpi=400)
            else:
                plt.savefig('{}//Figures//{}-{}'.format(self.learner.svfl, feat, self.learner.svnm), dpi=400)
            plt.close()

    def plot_important_features(self, svnm=''):
        # Copy, sort, and clean up dataframe
        df = self.learner.feature_importance_df.copy()
        df.sort_values(by='Feature Importance', inplace=True, ascending=False)
        df.rename(index={'reactor_number':'Reactor', 'temperature':'Temperature','space_velocity':'Space Velocity',
                         'n_elements':'Number of Elements', 'ammonia_concentration':'Ammonia Concentration',
                         }, inplace=True)

        pltdf = pd.DataFrame()

        for idx in df.index.values:
            spidx = idx.split('_')

            df.loc[idx, 'Feature'] = spidx[0]

            if len(spidx) == 2:
                df.loc[idx, 'Type'] = spidx[1]
                pltdf.loc[spidx[0], spidx[1]] = df.loc[idx, 'Feature Importance']
            else:
                df.loc[idx, 'Type'] = 'N/A'
                pltdf.loc[spidx[0], 'N/A'] = df.loc[idx, 'Feature Importance']

        f, ax = plt.subplots(figsize=(8,20))
        pltdf.plot(kind='barh', stacked='True', legend=True, ax=ax, color=sns.color_palette('muted', 3))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('{}//Figures//features-{}{}'.format(self.learner.svfl, self.learner.svnm, svnm), dpi=400)
        plt.close()

        f, ax = plt.subplots()
        pltdf.iloc[:10].plot(kind='barh', stacked='True', legend=True, ax=ax, color=sns.color_palette('muted', 3))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('{}//Figures//top10-{}{}'.format(self.learner.svfl, self.learner.svnm, svnm), dpi=400)
        plt.close()




        # sns.set_palette('plasma')

# TODO Implement Bokeh
    def bokeh_predictions(self, svnm=None):
        """ Comment """
        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@temperature')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.learner.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(self.graphdf)

        p.circle("Predicted Conversion", "Measured Conversion", size=12, source=source,
                 color='clr', line_color="black", fill_alpha=0.8)
        if svnm is None:
            output_file("{}\\htmls\\{}.html".format(self.learner.svfl, self.learner.svnm), title="stats.py")
        else:
            output_file("{}\\htmls\\{}.html".format(self.learner.svfl, svnm), title="stats.py")
        save(p)

    def bokeh_by_elements(self):
        """ HTML with overview with colorscheme that is per-element """
        if self.predictions is None:
            self.predict_crossvalidate()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@temperature')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        self.plot_df['bokeh_color'] = self.plot_df['Ele2_hues'].apply(rgb2hex)

        source = ColumnDataSource(self.plot_df)

        p.circle("Predicted Conversion", "Measured Conversion", size=12, source=source,
                 color='bokeh_color', line_color="black", fill_alpha=0.8)

        output_file("{}\\htmls\\{}_byeles.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def bokeh_averaged(self, whiskers=False):
        """ Comment """
        if self.predictions is None:
            self.predict_crossvalidate()

        df = pd.DataFrame(np.array([
            [int(nm.split('_')[0]) for nm in self.slave_dataset.index.values],
            self.predictions,
            self.labels_df.values,
            self.slave_dataset.loc[:, 'temperature'].values]).T,
                          columns=['ID', 'Predicted', 'Measured', 'Temperature'])

        cat_eles = self.slave_dataset.loc[:, 'Element Dictionary']
        vals = [''.join('{}({})'.format(key, str(int(val))) for key, val in x) for x in cat_eles]
        df['Name'] = vals

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('T', '@Temperature')
        ])
        tools.append(hover)

        unique_temps = len(df['Temperature'].unique())
        max_temp = df['Temperature'].max()
        min_temp = df['Temperature'].min()

        if max_temp == min_temp:
            df['color'] = pals.plasma(5)[4]
        else:
            pal = pals.plasma(unique_temps + 1)
            df['color'] = [pal[i]
                           for i in [int(unique_temps * (float(x) - min_temp) / (max_temp - min_temp))
                                     for x in df['Temperature']]]

        unique_names = np.unique(df.loc[:, 'Name'].values)

        final_df = pd.DataFrame()

        for nm in unique_names:
            nmdf = df.loc[df.loc[:, 'Name'] == nm]
            unique_temp = np.unique(nmdf.loc[:, 'Temperature'].values)

            for temperature in unique_temp:
                tdf = nmdf.loc[nmdf.loc[:, 'Temperature'] == temperature]
                add_df = tdf.iloc[0, :].copy()
                add_df['Measured'] = tdf['Measured'].mean()
                add_df['Measured Standard Error'] = tdf['Measured'].sem()
                add_df['Upper'] = tdf['Measured'].mean() + tdf['Measured'].sem()
                add_df['Lower'] = tdf['Measured'].mean() - tdf['Measured'].sem()
                add_df['n Samples'] = tdf['Measured'].count()

                final_df = pd.concat([final_df, add_df], axis=1)

        df = final_df.transpose()

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = Range1d(0,1)
        p.y_range = Range1d(0,1)
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = "Predicted Conversion"
        p.yaxis.axis_label = "Measured Conversion"
        p.grid.grid_line_color = "white"

        source = ColumnDataSource(df)

        p.circle("Predicted", "Measured", size=8, source=source,
                 color='color', line_color="black", fill_alpha=0.8)

        if whiskers:
            p.add_layout(
                Whisker(source=source, base="Predicted", upper="Upper", lower="Lower", level="overlay")
            )

        output_file("{}\\{}_avg.html".format(self.svfl, self.svnm), title="stats.py")
        save(p)

    def bokeh_important_features(self, svtag='IonEn',
                                 xaxis='Measured Conversion', xlabel='Measured Conversion', xrng=None,
                                 yaxis='Predicted Conversion', ylabel='Predicted Conversion', yrng=None,
                                 caxis='temperature'
                                 ):
        """ Comment """

        # uniqvals = np.unique(self.plot_df[caxis].values)
        # for cval in uniqvals:
        #     slice = self.plot_df[caxis] == cval
        #     plt.scatter(x=self.plot_df.loc[slice, xaxis], y=self.plot_df.loc[slice, yaxis],
        #                 c=self.plot_df.loc[slice, '{}_hues'.format(caxis)], label=cval, s=30, edgecolors='k')

        # unique_temps = len(featdf['temperature'].unique())
        # max_temp = featdf['temperature'].max()
        # min_temp = featdf['temperature'].min()
        #
        # if max_temp == min_temp:
        #     featdf['color'] = pals.plasma(5)[4]
        # else:
        #     pal = pals.plasma(unique_temps + 1)
        #     featdf['color'] = [pal[i]
        #                    for i in [int(unique_temps * (float(x) - min_temp) / (max_temp - min_temp))
        #                              for x in featdf['temperature']]]
        #
        # if temp_slice is not None:
        #     featdf = featdf[featdf['temperature'] == temp_slice]
        #
        if xrng is None:
            xrng = DataRange1d()
        if yrng is None:
            yrng = DataRange1d()

        tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
        hover = HoverTool(tooltips=[
            ('Name', '@Name'),
            ("ID", "@ID"),
            ('IonEn', '@IonizationEnergies_2_1')
        ])

        tools.append(hover)

        p = figure(tools=tools, toolbar_location="above", logo="grey", plot_width=600, plot_height=600, title=self.svnm)
        p.x_range = xrng
        p.y_range = yrng
        p.background_fill_color = "#dddddd"
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        p.grid.grid_line_color = "white"

        try:
            self.plot_df['bokeh_color'] = self.plot_df['{}_hues'.format(caxis)].apply(rgb2hex)
        except KeyError:
            self.plot_df['bokeh_color'] = 'blue'

        source = ColumnDataSource(self.plot_df)
        p.circle(xaxis, yaxis, size=12, source=source,
                 color='bokeh_color', line_color="black", fill_alpha=0.8)

        output_file("{}\\{}{}.html".format(self.svfl, self.svnm, '-{}'.format(svtag) if svtag is not '' else ''), title="stats.py")
        save(p)

