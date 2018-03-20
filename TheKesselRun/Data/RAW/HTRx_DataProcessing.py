import pandas as pd
import numpy as np
import re
import glob

# Whoever designed the code that generates this data...  grr...


def load_info_sheet(pth, thresh=15, sheetname='Info', skip_footer=13, skiprow=1):
    df = pd.read_excel(pth, skiprows=skiprow, skip_footer=skip_footer, sheet_name=sheetname)
    df.dropna(inplace=True, axis=1, thresh=thresh)
    df = df.loc[~np.isnan(df['Reactor'])]
    df.dropna(inplace=True, axis=0, thresh=5)
    return df


def load_activity_sheet(pth, sheet, cols=None):
    df = pd.read_excel(pth,
                       sheet_name=sheet,
                       usecols=cols)
    df.dropna(inplace=True, how='any')
    return df


def extract_activity_information(df):
    act_list = list()

    for tmp in df.values:
        if tmp[0] == 'Sample':
            continue

        reactor = tmp[0].split('_')[-1].split('.')[0].replace('R', '')
        type = 'down' if 'd' in tmp[0].split('-')[2] else 'up'
        temp = tmp[0].split('_')[0][-8:].split('-')[1][:3]
        pred = tmp[1]

        act_list += [[reactor, type, temp, pred]]

    return pd.DataFrame(act_list, columns=['Reactor', 'Ramp Direction', 'Temperature', 'Pred Value'])


def split_katies_ID(df):
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
                      index=['KatieID', 'Catalyst ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3']).T
         for vals in output_list],
        ignore_index=True
    )


def merge_ids_flows(flowsdf, iddf):
    df = pd.concat([flowsdf.set_index('Catalyst'), iddf.set_index('KatieID')], axis=1, join='inner')
    return df


def merge_activity_catalysts(actdf, catdf, nh3scale=100):
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
    output_df['NH3'] = output_df['NH3'] * nh3scale
    output_df['Concentration'] = (output_df['NH3'] - output_df['Pred Value']) / (output_df['NH3'])
    output_df.loc[output_df['Concentration'] < 0,  'Concentration'] = 0

    # output_df.drop(['Reactor(2)','KTag'] ,inplace=True)
    output_df = output_df[['ID', 'Ele1', 'Wt1', 'Ele2', 'Wt2', 'Ele3', 'Wt3',
                           'Reactor', 'Wt(g)', 'NH3', 'Space Velocity',
                           'Ramp Direction', 'Temperature', 'Pred Value', 'Concentration']]
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def read_v1_data():
    def read_spreadsheet_3():
        referencedata_df = load_info_sheet(r'.//NH3_v1_data_3.xlsx', thresh=10, sheetname='Redo')
        activitydata_df = load_activity_sheet(pth=r'./NH3_v1_data_3.xlsx', sheet='Data 25 Jan 2018', cols='A,E')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Processed//SS3.csv')

    def read_spreadsheet_4():
        referencedata_df = load_info_sheet(r'.//NH3_v1_data_4.xlsx', thresh=10)
        activitydata_df = load_activity_sheet(pth=r'./NH3_v1_data_4.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Processed//SS4.csv')

    def read_spreadsheet_5():
        referencedata_df = load_info_sheet(r'.//NH3_v1_data_5.xlsx', thresh=10, skip_footer=26)
        activitydata_df = load_activity_sheet(pth=r'./NH3_v1_data_5.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf)
        df.to_csv('..//Processed//SS5.csv')

    def read_spreadsheet_6():
        referencedata_df = load_info_sheet(r'.//NH3rxn6.xlsx', sheetname='Sheet1', thresh=10, skip_footer=26, skiprow=0)
        referencedata_df['Space Velocity'] = referencedata_df['Total (sccm)'].values / referencedata_df[
            'Weight'].values * 3.95
        activitydata_df = load_activity_sheet(pth=r'./NH3rxn6.xlsx', sheet='Raw', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
        df.to_csv('..//Processed//SS6.csv')

    def read_spreadsheet_7():
        referencedata_df = load_info_sheet(r'.//NH3rxn7.xlsx', sheetname='Info', thresh=10, skip_footer=10, skiprow=1)
        activitydata_df = load_activity_sheet(pth=r'./NH3rxn7.xlsx', sheet='Data', cols='A,D')

        actdf = extract_activity_information(activitydata_df)
        catdf = merge_ids_flows(referencedata_df, split_katies_ID(referencedata_df))

        df = merge_activity_catalysts(actdf, catdf, nh3scale=1)
        df.to_csv('..//Processed//SS7.csv')

    read_spreadsheet_3()
    read_spreadsheet_4()
    read_spreadsheet_5()
    read_spreadsheet_6()
    read_spreadsheet_7()


def read_v4_data():
    datpth = r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Data\RAW\NH3_v4_Data.xlsx'

    for num in [3, 4, 5, 6, 7]:
        # if num != 6:
        #     continue
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
        df.to_csv('..//Processed//SS{}.csv'.format(num))

def create_super_monster_file():
    pths = glob.glob('..//Processed//SS*.csv')

    df = pd.concat([pd.read_csv(pth, index_col=0) for pth in pths], ignore_index=True)
    df.to_csv('..//Processed//AllData.csv')


if __name__ == '__main__':
    read_v4_data()
    create_super_monster_file()