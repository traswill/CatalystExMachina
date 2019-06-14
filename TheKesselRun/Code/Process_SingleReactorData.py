import pandas as pd
import numpy as np
import re
import glob
import itertools

class SingleReactorData():
    def __init__(self, pth, T_split_loc, id, eles, wts, cat_wt, sv, nh3, pres):
        self.gc_df = pd.DataFrame()
        self.conv_df = pd.DataFrame()

        self.read_data(pth=pth, T_split_loc=T_split_loc)
        self.add_reactor_information(sv, nh3, pres)
        self.add_catalyst_information(id, eles, wts, cat_wt)
        self.add_group()
        self.conv_df['Reactor'] = -1

    def read_data(self, pth, T_split_loc=1):
        df = pd.read_csv(pth, delimiter='\t',  header=None, names=np.linspace(0,14,15))
        df = df.loc[(df[0.0] == 'Data File Name') | (df[0.0] == '1') | (df[0.0] == '2')]
        print(df.loc[df[0.0] == 'Data File Name'].loc[:, 1.0].values)
        temps = [x.split('\\')[-1].split('_')[T_split_loc].replace('C', '') for x in
                 df.loc[df[0.0] == 'Data File Name'].loc[:, 1.0].values]
        h2_conc = df.loc[df[1.0] == 'Hydrogen'].loc[:, 5.0].values.astype(float).tolist()
        n2_conc = df.loc[df[1.0] == 'Nitrogen'].loc[:, 5.0].values.astype(float).tolist()
        self.gc_df = pd.DataFrame([temps, h2_conc, n2_conc], index=['Temperature', 'Hydrogen', 'Nitrogen'],
                                 dtype=float).T

        self.calc_conversion()
        self.average_conversion()

    def calc_conversion(self, ammonia_concentration=100):
        extent = (ammonia_concentration * self.gc_df['Hydrogen'] / 100) / ((3 / 2) - self.gc_df['Hydrogen'] / 100)
        self.gc_df['Conversion'] = 1 - ((ammonia_concentration - extent) / (ammonia_concentration + extent))

    def average_conversion(self):
        self.conv_df = self.gc_df.groupby(['Temperature']).mean()
        self.conv_df['StDev'] = self.gc_df.groupby(['Temperature']).std().loc[:, 'Conversion']
        self.conv_df.loc[self.conv_df['Conversion'] < 0, 'Conversion'] = 0
        self.conv_df.drop(columns=['Hydrogen', 'Nitrogen'], inplace=True)
        self.conv_df.reset_index(inplace=True)

    def add_catalyst_information(self, id, eles, wts, cat_wt):
        for idx, ele in enumerate(eles):
            self.conv_df['Ele{}'.format((idx+1))] = ele
            self.conv_df['Wt{}'.format((idx + 1))] = wts[idx]

        self.conv_df['ID'] = id
        self.conv_df['Wt(g)'] = cat_wt

    def add_reactor_information(self, sv, nh3, pres):
        self.conv_df['Space Velocity'] = sv
        self.conv_df['NH3'] = nh3
        self.conv_df['Pressure'] = pres

    def add_group(self):
        # Group catalysts by product of atomic numbers in catalyst - i.e. RuFeK is 44*26*19 = 21736
        ele_dict = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation', 'Atomic Number'],
                               index_col='Abbreviation').transpose().to_dict(orient='list')

        group = ele_dict.get(self.conv_df.loc[0, 'Ele1'])[0] * \
                ele_dict.get(self.conv_df.loc[0, 'Ele2'])[0] * \
                ele_dict.get(self.conv_df.loc[0, 'Ele3'])[0]

        self.conv_df['Groups'] = group


def compile_single_reactor_data():
    compiled_df = pd.DataFrame()

    # RuYK Catalysts
    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(2)1312 RuYK\5bar 19.10sccm\(2)1312 RuYK T-data.txt",
        T_split_loc=1,
        id='31(2)',
        eles=['Ru', 'Y', 'K'],
        wts=[1, 3, 12],
        cat_wt=0.200,
        sv=19,
        nh3=100,
        pres=5
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(2)1312 RuYK\5200 ml NH3 gcat hr 43 sccm\ASCIIData002.txt",
        T_split_loc=1,
        id='31(2)',
        eles=['Ru', 'Y', 'K'],
        wts=[1, 3, 12],
        cat_wt=0.503,
        sv=43,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(2)2212 RuYK\SV 2 hr-1\2212 RuYK 2hr-1.txt",
        T_split_loc=1,
        id='30(2)',
        eles=['Ru', 'Y', 'K'],
        wts=[2, 2, 12],
        cat_wt=0.200,
        sv=22.83,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(2)3112 RuYK\RuYK litcompare\29(2) RuYK.txt",
        T_split_loc=1,
        id='29(2)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=43,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(2)3112 RuYK\SV 2 hr-1\3112 RuYK 2hr-1.txt",
        T_split_loc=1,
        id='29(2)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=22.83,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(3)3112 RuYK\5200 mlNH3-hr-gcat\(3)RuYK litcompare.txt",
        T_split_loc=1,
        id='29(3)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=43,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(3)3112 RuYK\5200 mlNH3-hr-gcat at 22 sccm\ASCIIData001.txt",
        T_split_loc=1,
        id='29(3)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=22,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(5)3112 RuYK\5200 ml NH3 500mg 1bar\ASCIIData.txt",
        T_split_loc=1,
        id='29(4)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.5036,
        sv=43.6,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\(5u)3112 RuYK\5200 ml Nh3 1bar\ASCIIData.txt",
        T_split_loc=1,
        id='29(5u)',
        eles=['Ru', 'Y', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.505,
        sv=43.8,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\1 Ru\0.199g catalyst 17.2 scm NH3 5200 ml Nh3\ASCIIData.txt",
        T_split_loc=1,
        id='143',
        eles=['Ru', '-', '--'],
        wts=[1, 0, 0],
        cat_wt=0.199,
        sv=17.2,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3 Ru\ASCIIData.txt",
        T_split_loc=1,
        id='141',
        eles=['Ru', '-', '--'],
        wts=[3, 0, 0],
        cat_wt=0.502,
        sv=17.2,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\1312 RuCuK\ASCIIData.txt",
        T_split_loc=1,
        id='17',
        eles=['Ru', 'Cu', 'K'],
        wts=[1, 3, 12],
        cat_wt=0.500,
        sv=43,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\1312 RuHfK\5bar 47.35sccm\ASCIIData.txt",
        T_split_loc=1,
        id='60',
        eles=['Ru', 'Hf', 'K'],
        wts=[1, 3, 12],
        cat_wt=0.500,
        sv=47.35,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3112 RuBiK\ASCIIData.txt",
        T_split_loc=1,
        id='76',
        eles=['Ru', 'Bi', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=47,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3112 RuMgK\RuMgK LitCompare\RuMgK litcompare001.txt",
        T_split_loc=1,
        id='33',
        eles=['Ru', 'Mg', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=47,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3112 RuNiK\Pure NH3 5200 ml\ASCIIData.txt",
        T_split_loc=1,
        id='40',
        eles=['Ru', 'Ni', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=47,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3112 RuScK\61 3112 RuScK litcompare\61 3112 RuScK litcompare.txt",
        T_split_loc=1,
        id='61',
        eles=['Ru', 'Sc', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=47,
        nh3=100,
        pres=1
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3112 RuSrK\3112 RuSrK litcompare\RuSrK litcompare001.txt",
        T_split_loc=1,
        id='73',
        eles=['Ru', 'Sr', 'K'],
        wts=[3, 1, 12],
        cat_wt=0.500,
        sv=47,
        nh3=100,
        pres=1
    ).conv_df])

    # MAY BE BAD DATA
    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3120 RuMgCs\19 sccm NH3 5 bar r2\ASCIIData.txt",
        T_split_loc=2,
        id='211',
        eles=['Ru', 'Mg', 'Cs'],
        wts=[3, 1, 20],
        cat_wt=0.2017,
        sv=19,
        nh3=100,
        pres=5
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3125 RuHfCs\213 5 bar 19 sccm\Tsweep data.txt",
        T_split_loc=2,
        id='213',
        eles=['Ru', 'Hf', 'Cs'],
        wts=[3, 1, 25],
        cat_wt=0.2017,
        sv=19,
        nh3=100,
        pres=5
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3125 RuMgK\19 sccm NH3 5 bar r2\ASCIIData.txt",
        T_split_loc=2,
        id='198',
        eles=['Ru', 'Mg', 'K'],
        wts=[3, 1, 25],
        cat_wt=0.200,
        sv=19,
        nh3=100,
        pres=5
    ).conv_df])

    compiled_df = pd.concat([compiled_df, SingleReactorData(
        pth=r"C:\Users\quick\OneDrive - University of South Carolina\Data\Raw - GC Data\3125 RuYCs\19 sccm 5 bar\ASCIIData.txt",
        T_split_loc=2,
        id='198',
        eles=['Ru', 'Y', 'Cs'],
        wts=[3, 1, 25],
        cat_wt=0.197,
        sv=19,
        nh3=100,
        pres=5
    ).conv_df])

    compiled_df.dropna(inplace=True)
    compiled_df.to_csv(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Data\Processed\Single Reactor Data.csv')

if __name__ == '__main__':
    compile_single_reactor_data()