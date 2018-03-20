import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('.//AllData.csv',  index_col=0)

    uniid = np.unique(df.loc[:, 'ID'].values)

    final_df = pd.DataFrame()

    for id in uniid:
        iddf = df.loc[df.loc[:, 'ID'] == id]
        unitemps = np.unique(iddf.loc[:, 'Temperature'].values)
        for temp in unitemps:
            tempdf = iddf.loc[iddf.loc[:, 'Temperature'] == temp]

            add_df = tempdf.iloc[0, :].copy()
            add_df['Concentration'] = tempdf['Concentration'].mean()
            add_df['Standard Error'] = tempdf['Concentration'].sem()
            add_df['nAveraged'] = tempdf['Concentration'].count()
            add_df.drop(index=['Ramp Direction','Pred Value'], inplace=True)

            final_df = pd.concat([final_df, add_df], axis=1)

    final_df = final_df.transpose()[['ID','Ele1','Wt1','Ele2','Wt2','Ele3','Wt3','Reactor','NH3',
                                     'Space Velocity','Temperature','Concentration','Standard Error','nAveraged']]

    final_df.to_csv('.//AllData_Condensed.csv')
