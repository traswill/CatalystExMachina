import pandas as pd
import numpy as np
import re
import glob
import itertools

# Whoever designed the code that generates this data...  grr...


def load_info_sheet(pth, thresh=15, sheetname='Info', skip_footer=13, skiprow=1):
    """ Reads data from path, drops NAN columns up to a threshold (variable per sheet), trashes empty reactor columns,
    and returns a dataframe """

    df = pd.read_excel(pth, skiprows=skiprow, skipfooter=skip_footer, sheet_name=sheetname)
    df.dropna(inplace=True, axis=1, thresh=thresh)
    df = df.loc[~np.isnan(df['Reactor'])]
    df.dropna(inplace=True, axis=0, thresh=5)
    return df


def load_activity_sheet(pth, sheet, cols=None):
    """ Literally just loads it and drops NANs """
    df = pd.read_excel(pth,
                       sheet_name=sheet,
                       usecols=cols)
    df.dropna(inplace=True, how='any')
    return df


def extract_activity_information(df, proc='default'):
    """
    Input: Activity datasheet in the form of a dataframe
    Output: Dataframe with 4 columns, numerical index

    Regex search is used to parse file name in spreadsheet to determine if the ramp direction is up or down
    Format = [000d.*-0]
    """
    act_list = list()

    for tmp in df.values:
        if tmp[0] == 'Sample':
            continue

        if proc == 'default':
            reactor = tmp[0].split('_')[-1].split('.')[0].replace('R', '')
            type = 'down' if re.search('\d{3}d.*-\d',tmp[0]) is not None else 'up'
            temp = tmp[0].split('_')[0][-8:].split('-')[1][:3]
            pred = tmp[1]
        elif proc == 'data12':
            reactor = tmp[0].split('_')[-1].split('.')[0].replace('R', '')
            type = 'down' if re.search('\d{3}d', tmp[0]) is not None else 'up'
            temp = tmp[0].split('_')[2].replace('C', '')
            pred = tmp[1]
        else:
            print('Proc not recognized. Resort to default.')
            reactor = tmp[0].split('_')[-1].split('.')[0].replace('R', '')
            type = 'down' if re.search('\d{3}d.*-\d', tmp[0]) is not None else 'up'
            temp = tmp[0].split('_')[0][-8:].split('-')[1][:3]
            pred = tmp[1]

        act_list += [[reactor, type, temp, pred]]

    return pd.DataFrame(act_list, columns=['Reactor', 'Ramp Direction', 'Temperature', 'Pred Value'])



def split_katies_ID(df):
    """
    This code parses Katie's IDs of the form "1,3,12 RuFeK" into a dataframe with columns:
        CatalystObject Index:     ##
        Concentration:      [1, 3, 12]
        Elements:           [Ru, Fe, K]
        CatalystObject Object:    [*, *, *] (These are the objects from class CatalystObject)

    The code has been modified to especially handle '-' and '--' values, which are placeholders for monometallic and
    bimetallic catalysts.  A '-' replaces the second element in a trimetallic, whereas a '--' replaces the third metal,
    typically K.

    """
    output_list = list()

    for catdat in df['Catalyst']:
        catlst = catdat.split(' ')
        if len(catlst) == 3:
            cat_index = int(catlst[0])
            catconc = catlst[1]
            cateles = catlst[2]

            cateles = re.findall('[A-Z][^A-Z]*', cateles)
            n_eles = len(cateles)

            if n_eles == 1:
                catconc = [catconc, 0, 0]
                cateles = [cateles[0], '-', '--']
            elif n_eles == 2:
                catconc = [catconc[0], catconc[1:], 0]
                cateles = [cateles[0], cateles[1], '--']
            elif n_eles == 3:
                if '.' in catconc:
                    catconc = [catconc[0], catconc[1:4], catconc[4:]]
                else:
                    catconc = [catconc[0], catconc[1], catconc[2:]]

            catconc = np.array(catconc, dtype=float)

            output_list += [[cat_index, catconc, cateles, catdat]]

    return pd.concat(
        [pd.DataFrame([vals[3], vals[0],
                       vals[2][0], vals[1][0],
                       vals[2][1], vals[1][1],
                       vals[2][2], vals[1][2]],
                      index=['KatieID', 'ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3']).T
         for vals in output_list],
        ignore_index=True
    )


def merge_ids_flows(flowsdf, iddf):
    """ Concatinate data with parsed elements and concentrations from Katie's IDs """
    df = pd.concat([flowsdf.set_index('Catalyst'), iddf.set_index('KatieID')], axis=1, join='inner')
    return df


def merge_activity_catalysts(actdf, catdf, nh3scale=100):
    """

    :param actdf:
    :param catdf:
    :param nh3scale: Scale of the NH3 concentration in the raw files.
    :return:
    """
    final_cols = ['Reactor', 'Wt(g)', 'Tot(SCCM)', 'HE(SCCM)', 'NH3', 'Space Velocity',
                  'ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3',
                  'Reactor(2)', 'Ramp Direction', 'Temperature', 'Pred Value']

    output_df = pd.DataFrame(columns=final_cols)

    # Append Activity Results to Output DF
    for rct in catdf.loc[:, 'Reactor'].values:
        if rct not in np.array(actdf['Reactor'].values, dtype=float).tolist():
            continue

        df_slice = actdf[np.array(actdf['Reactor'].values, dtype=float) == rct]
        output_slice = catdf[np.array(catdf['Reactor'].values, dtype=float) == rct]

        df = pd.concat([pd.concat([output_slice.iloc[0], row], axis=0) for index, row in df_slice.iterrows()], axis=1).T
        df.columns = final_cols
        output_df = pd.concat([output_df, df])

    output_df.loc[output_df['Pred Value'] < 0, 'Pred Value'] = 0

    # Use nominal NH3 concentration
    output_df['NH3'] = output_df['NH3'] * nh3scale
    output_df.loc[output_df['NH3'] < 2, 'NH3'] = 1
    output_df['Conversion'] = (1 - output_df['Pred Value']) / 1

    # Use reported NH3 concentration
    # output_df['NH3'] = output_df['NH3'] * nh3scale
    # output_df['Conversion'] = (output_df['NH3'] - output_df['Pred Value']) / (output_df['NH3'])

    output_df.loc[output_df['Conversion'] < 0,  'Conversion'] = 0

    # output_df.drop(['Reactor(2)','KTag'] ,inplace=True)
    output_df = output_df[['ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3',
                           'Reactor', 'Wt(g)', 'NH3', 'Space Velocity',
                           'Ramp Direction', 'Temperature', 'Pred Value', 'Conversion']]
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def read_v1_data():
    def read_spreadsheet_3():
        referencedata_df = load_info_sheet('..//RAW//NH3_v1_data_3.xlsx', thresh=10, sheetname='Redo')
        activitydata_df = load_activity_sheet(pth='..//RAW//NH3_v1_data_3.xlsx', sheet='Data 25 Jan 2018', cols='A,E')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Data//Processed//SS3.csv')

    def read_spreadsheet_4():
        referencedata_df = load_info_sheet('..//RAW//NH3_v1_data_4.xlsx', thresh=10)
        activitydata_df = load_activity_sheet(pth='..//RAW/NH3_v1_data_4.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Data//Processed//SS4.csv')

    def read_spreadsheet_5():
        referencedata_df = load_info_sheet('..//RAW//NH3_v1_data_5.xlsx', thresh=10, skip_footer=26)
        activitydata_df = load_activity_sheet(pth='..//RAW/NH3_v1_data_5.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Data//Processed//SS5.csv')

    def read_spreadsheet_6():
        referencedata_df = load_info_sheet('..//RAW//NH3rxn6.xlsx', sheetname='Sheet1', thresh=10, skip_footer=26, skiprow=0)
        referencedata_df['Space Velocity'] = referencedata_df['Total (sccm)'].values / referencedata_df[
            'Weight'].values * 3.95
        activitydata_df = load_activity_sheet(pth='..//RAW/NH3rxn6.xlsx', sheet='Raw', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
        df.to_csv('..//Data//Processed//SS6.csv')

    def read_spreadsheet_7():
        referencedata_df = load_info_sheet('..//RAW//NH3rxn7.xlsx', sheetname='Info', thresh=10, skip_footer=10, skiprow=1)
        activitydata_df = load_activity_sheet(pth='..//RAW/NH3rxn7.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
        df.to_csv('..//Data//Processed//SS7.csv')

    read_spreadsheet_3()
    read_spreadsheet_4()
    read_spreadsheet_5()
    read_spreadsheet_6()
    read_spreadsheet_7()


def read_v4_data():
    datpth = r'..\Data\RAW\NH3_v4_Data.xlsx'

    for num in [3, 4, 5, 6, 7]:
        if num == 3:
            nh3scale = 1
            referencedata_df = load_info_sheet(datpth, sheetname='Info {}'.format(num), thresh=15, skip_footer=15,
                                               skiprow=1)
        elif num == 4:
            nh3scale = 1
            referencedata_df = load_info_sheet(datpth, sheetname='Info {}'.format(num), thresh=8, skip_footer=15,
                                               skiprow=1)
        elif num == 5:
            nh3scale = 1
            referencedata_df = load_info_sheet(datpth, sheetname='Info {}'.format(num), thresh=8, skip_footer=15,
                                               skiprow=1)
        elif num == 6:
            nh3scale = 100
            referencedata_df = load_info_sheet(datpth, sheetname='Info {}'.format(num), thresh=10, skip_footer=26,
                                               skiprow=0)
            referencedata_df['Space Velocity'] = referencedata_df['Total (sccm)'].values / \
                                                 referencedata_df['Weight'].values * 3.95
        elif num == 7:
            nh3scale = 100
            referencedata_df = load_info_sheet(datpth, sheetname='Info {}'.format(num), thresh=12, skip_footer=10,
                                               skiprow=1)
        else:
            nh3scale = 1

        activitydata_df = load_activity_sheet(datpth, sheet='RXN {}'.format(num), cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf, nh3scale=nh3scale)
        df.to_csv('..//Data//Processed//SS{}.csv'.format(num))

def read_v4_data_8():
    datpth = r'..\Data\RAW\NH3_v4_data_8.xlsx'
    referencedata_df = load_info_sheet(datpth, sheetname='Info', thresh=12, skip_footer=45, skiprow=1)
    activitydata_df = load_activity_sheet(datpth, sheet='Data', cols='A,D')

    actdf = extract_activity_information(activitydata_df)

    # Katie said to drop the down ramp because they were exposed to a higher concentration of NH3
    actdf.drop(actdf[actdf.loc[:, 'Ramp Direction'] == 'down'].index, inplace=True)
    catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

    df = merge_activity_catalysts(actdf, catdf, nh3scale=100)
    df.to_csv('..//Data//Processed//SS8.csv')

def read_data_9():
    datpth = r'..\Data\RAW\NH3_v9_data_9.xlsx'
    referencedata_df = load_info_sheet(datpth, sheetname='Info', thresh=12, skip_footer=0, skiprow=1)
    activitydata_df = load_activity_sheet(datpth, sheet='Data', cols='A,D')

    actdf = extract_activity_information(activitydata_df)

    # Katie said to drop the down ramp because they were exposed to a higher concentration of NH3
    actdf.drop(actdf[actdf.loc[:, 'Ramp Direction'] == 'down'].index, inplace=True)

    catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

    df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
    df.to_csv('..//Data//Processed//SS9.csv')

def read_data_9_updated():
    datpth = r'..\Data\RAW\NH3_v9_data_9_updated.xlsx'
    referencedata_df = load_info_sheet(datpth, sheetname='Info', thresh=12, skip_footer=12, skiprow=1)
    activitydata_df = load_activity_sheet(datpth, sheet='Data 10%', cols='A,D,G')

    act_list = list()

    for tmp in activitydata_df.values:
        if tmp[0] == 'Sample':
            continue
        reactor = tmp[0].split('_')[-1].split('.')[0].replace('R', '')
        temp = 350
        pred = tmp[1]
        ghsv = tmp[2]

        act_list += [[reactor, temp, pred, ghsv]]

    actdf = pd.DataFrame(act_list, columns=['Reactor', 'Temperature', 'Pred Value', 'GHSV'])
    catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

    catdf.drop(columns=['Total (sccm)', 'Ar (sccm)', 'NH3 actual (%)', 'Space Velocity'], inplace=True)

    output_df = pd.DataFrame()

    # Append Activity Results to Output DF
    for rct in catdf.loc[:, 'Reactor'].values:
        if rct not in np.array(actdf['Reactor'].values, dtype=float).tolist():
            continue

        df_slice = actdf[np.array(actdf['Reactor'].values, dtype=float) == rct]
        output_slice = catdf[np.array(catdf['Reactor'].values, dtype=float) == rct]

        df = pd.concat([pd.concat([output_slice.iloc[0], row], axis=0) for index, row in df_slice.iterrows()], axis=1).T
        output_df = pd.concat([output_df, df])

    output_df.loc[output_df['Pred Value'] < 0, 'Pred Value'] = 0
    output_df['NH3'] = 5
    output_df['Conversion'] = (output_df['NH3'] - output_df['Pred Value']) / (output_df['NH3'])
    output_df.loc[output_df['Conversion'] < 0, 'Conversion'] = 0

    output_df.columns = ['Reactor', 'Wt(g)', 'ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3',
                         'Reactor', 'Temperature', 'Pred Value', 'Space Velocity', 'NH3', 'Conversion']

    output_df['Ramp Direction'] = np.nan

    output_df = output_df[['ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3',
                           'Reactor', 'Wt(g)', 'NH3', 'Space Velocity',
                           'Ramp Direction', 'Temperature', 'Pred Value', 'Conversion']]
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv('..//Data//Processed//SS9_u.csv')

def read_data_12():
    datpth = r"C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Data\RAW\NH3_v9_data_12.xlsx"
    referencedata_df = load_info_sheet(datpth, sheetname='Info', thresh=10, skip_footer=14, skiprow=1)
    activitydata_df = load_activity_sheet(datpth, sheet='Data', cols='A,D')

    actdf = extract_activity_information(activitydata_df, proc='data12')

    # Katie said to drop the down ramp because they were exposed to a higher concentration of NH3
    actdf.drop(actdf[actdf.loc[:, 'Ramp Direction'] == 'down'].index, inplace=True)

    catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

    df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
    df.to_csv('..//Data//Processed//SS12.csv')

def create_super_monster_file():
    pths = glob.glob('..//Data//Processed//SS*.csv')

    df = pd.concat([pd.read_csv(pth, index_col=0) for pth in pths], ignore_index=True, sort=True)
    df.to_csv('..//Data//Processed//AllData.csv')

    uniid = np.unique(df.loc[:, 'ID'].values)

    final_df = pd.DataFrame()

    for id in uniid:
        iddf = df.loc[df.loc[:, 'ID'] == id]
        unitemps = np.unique(iddf.loc[:, 'Temperature'].values)
        uniammonia = np.unique(iddf.loc[:, 'NH3'].values)
        unispacevelocity = np.unique(iddf.loc[:, 'Space Velocity'].values)

        lists = [unitemps, uniammonia, unispacevelocity]
        slice_list = list(itertools.product(*lists))

        for temp, ammo, ghsv in slice_list:
            temporary_df = iddf.loc[(iddf.loc[:, 'Temperature'] == temp) &
                                    (iddf.loc[:, 'NH3'] == ammo) &
                                    (iddf.loc[:, 'Space Velocity'] == ghsv)]

            if temporary_df.empty:
                continue

            add_df = temporary_df.iloc[0, :].copy()
            add_df['Conversion'] = temporary_df['Conversion'].mean()
            add_df['Standard Error'] = temporary_df['Conversion'].sem()
            add_df['nAveraged'] = temporary_df['Conversion'].count()

            final_df = pd.concat([final_df, add_df], axis=1, sort=True)

    final_df = final_df.transpose()[['ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3', 'Reactor', 'NH3',
                                     'Space Velocity', 'Temperature', 'Conversion', 'Standard Error', 'nAveraged']]

    # Group catalysts by product of atomic numbers in catalyst - i.e. RuFeK is 44*26*19 = 21736
    ele_dict = pd.read_csv('..\\Data\\Elements.csv', usecols=['Abbreviation', 'Atomic Number'], index_col='Abbreviation').transpose().to_dict(orient='list')
    # groups = [ele_dict.get(x)[0] for x in final_df['Ele2'].values] # Old grouping method using second element atomic number only
    groups = [ele_dict.get(x)[0] * ele_dict.get(y)[0] * ele_dict.get(z)[0]
              for x, y, z in final_df[['Ele1', 'Ele2', 'Ele3']].values]

    final_df['Groups'] = groups

    final_df.to_csv('..//Data//Processed//AllData_Condensed.csv')

def read_data0_data1():
    pth = r'../Data/RAW/NH3_v0_data1_data2.xlsx'
    df = pd.read_excel(pth)
    df = df[['ID','Ele1','Wt1','Ele2','Wt2','Ele3','Wt3','Reactor','Weight','NH3','Space Velocity','Ramp Direction','Temperature','Pred Value','Conversion']]
    df.columns = ['ID','Ele1','Wt1','Ele2','Wt2','Ele3','Wt3','Reactor','Wt(g)','NH3','Space Velocity','Ramp Direction','Temperature','Pred Value','Conversion']
    df.to_csv('..//Data//Processed//SS1.csv')

def read_data_11():
    datpth = r"C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Data\RAW\NH3_v9_data_11.xlsx"
    referencedata_df = load_info_sheet(datpth, sheetname='Info', thresh=10, skip_footer=14, skiprow=1)
    activitydata_df = load_activity_sheet(datpth, sheet='Data', cols='A,D')

    actdf = extract_activity_information(activitydata_df, proc='default')

    # Katie said to drop the down ramp because they were exposed to a higher concentration of NH3
    actdf.drop(actdf[actdf.loc[:, 'Ramp Direction'] == 'down'].index, inplace=True)

    catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

    df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
    df.to_csv('..//Data//Processed//SS11.csv')

if __name__ == '__main__':
    read_data0_data1()
    read_v4_data()
    read_v4_data_8()
    read_data_9()
    read_data_9_updated()
    read_data_11()
    read_data_12()
    # exit()
    create_super_monster_file()