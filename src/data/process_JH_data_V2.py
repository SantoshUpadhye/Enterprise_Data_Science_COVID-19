import pandas as pd
import numpy as np

from datetime import datetime


def store_relational_JH_data():
    ''' Transformes the COVID data in a relational data set
    '''

    data_path='C:/ProgramData/Anaconda3/eps_covid19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')

    pd_relational_model.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))


def get_all_country_flat_table():
    ''' Creating a flat table having information of all countries' confirmed infected cases to be used for SIR modelling
    '''
    COVID_data_path='C:/ProgramData/Anaconda3/eps_covid19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df_data = pd.read_csv(COVID_data_path)

    df_data = df_data.drop(['Lat','Long'],axis=1)

    df_data = df_data.rename(columns={'Country/Region':'country',
                                   'Province/State':'state'})

    df_data['state'] = df_data['state'].fillna('no')

    full_country_list = df_data['country'].unique().tolist()

    time_idx = df_data.columns[2:]
    df_full_flat_table = pd.DataFrame({'date':time_idx})

    for each in full_country_list:
        df_full_flat_table[each] = np.array(df_data[df_data['country']==each].iloc[:,2::].sum(axis=0))

    time_idx=[datetime.strptime(each,"%m/%d/%y") for each in df_full_flat_table.date] # convert to datetime
    time_str=[each.strftime('%y-%m-%d') for each in time_idx] # convert back to date ISO norm (str)
    df_full_flat_table['date']=time_idx

    df_full_flat_table.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/all_country_flat_table.csv', sep=';',index=False)
    print(' Number of rows stored in all country flat table: '+str(df_full_flat_table.shape[0]))


if __name__ == '__main__':

    store_relational_JH_data()
    get_all_country_flat_table()
