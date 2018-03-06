import pandas as pd
import numpy as np
import re

# Whoever designed the code that generates this data...  grr...

def read_spreadsheet_3():
    # 1. Load Catalyst Information Page
    reference_df = pd.read_excel(r'./NH3rxn3_TW.xlsx', skiprows=1, sheet_name='Redo')
    reference_df.dropna(inplace=True, axis=1, thresh=15)
    reference_df = reference_df.loc[~np.isnan(reference_df['Reactor'])]

    # 2. Load and Handle Temperature Data
    # activity_df = pd.read_excel(r'./NH3rxn3_TW.xlsx',
    #                             sheet_name='Data 25 Jan 2018',
    #                             usecols='A,D:F')
    # activity_df.dropna(inplace=True, how='any')
    #
    # act_list = list()
    # for tmp in activity_df.values:
    #     reactor = tmp[0].split('_')[-1].split('.')[0].replace('R','')
    #     type = 'down' if 'd' in tmp[0].split('-')[2] else 'up'
    #     temp = tmp[1]
    #     pred = tmp[2]
    #     conc = tmp[3]
    #
    #     act_list += [[reactor, type, temp, pred, conc]]
    #
    # activity_df = pd.DataFrame(act_list, columns=['Reactor', 'Ramp Direction', 'Temperature', 'Pred Value', 'Concentration'])

    # 3. Handle Catalyst Information, Insert Activity
    output_list = list()
    for catdat in reference_df['Catalyst']:
        catlst = catdat.split(' ')
        if len(catlst) == 3:
            cat_index = int(catlst[0])
            catconc = catlst[1]
            cateles = catlst[2]

            cateles = re.findall('[A-Z][^A-Z]*', cateles)
            n_eles = len(cateles)

            if n_eles == 1:
                catconc = (catconc)
            elif n_eles == 3:
                if '.' in catconc:
                    catconc = [catconc[0], catconc[1:4], catconc[4:]]
                else:
                    catconc = [catconc[0],catconc[1],catconc[2:]]

            catconc = np.array(catconc, dtype=float)

            output_list += [[cat_index, catconc, cateles]]

    print(pd.DataFrame(output_list))
    print(reference_df)

if __name__ == '__main__':
    read_spreadsheet_3()