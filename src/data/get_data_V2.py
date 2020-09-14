import subprocess
import os

import pandas as pd
import numpy as np

from datetime import datetime

import requests
import json

from bs4 import BeautifulSoup


def get_johns_hopkins():

    ''' Get data by a git pull request, the source code has to be pulled first
        Result is stored in the predifined csv structure
    '''
    git_pull = subprocess.Popen(["git", "pull"], #"/usr/bin/git pull" ,
                        cwd = os.path.dirname( 'C:/ProgramData/Anaconda3/eps_covid19/data/raw/COVID-19/' ),
                        shell = True,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE )
    (out, error) = git_pull.communicate()


    print("Error : " + str(error))
    print("out : " + str(out))


def get_current_data_germany():
    ''' Get current data from germany, attention API endpoint not too stable
        Result data frame is stored as pd.DataFrame
    '''
    # 16 states
    #data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Coronaf%C3%A4lle_in_den_Bundesl%C3%A4ndern/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    # 400 regions / Landkreise
    data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    json_object=json.loads(data.content)
    full_list=[]
    for pos,each_dict in enumerate (json_object['features'][:]):
        full_list.append(each_dict['attributes'])

    pd_full_list=pd.DataFrame(full_list)
    pd_full_list.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/raw/NPGEO/GER_state_data.csv',sep=';')
    print(' Number of regions rows: '+str(pd_full_list.shape[0]))


def get_population_data():
    ''' Get population data of all countries, which is used in SIR model, by web scraping'''
    page = requests.get("https://www.worldometers.info/world-population/population-by-country/")
    soup = BeautifulSoup(page.content, 'html.parser')
    html_table_pop = soup.find('table')
    all_rows_pop = html_table_pop.find_all('tr')
    final_pop_data_list=[]
    for pos,rows in enumerate(all_rows_pop):
        col_list= [each_col.get_text(strip=True) for each_col in rows.find_all('td') ]
        final_pop_data_list.append(col_list)
    reqd_pop_list = pd.DataFrame(final_pop_data_list).dropna()\
                    .rename(columns={1:'country', 2:'population'})
    reqd_pop_list = reqd_pop_list[['country','population']]
    reqd_pop_list["country"]= reqd_pop_list["country"].replace({'Myanmar':'Burma', 'Czech Republic (Czechia)': 'Czechia', 'DR Congo': 'Congo (Kinshasa)', 'Congo': 'Congo (Brazzaville)', 'South Korea': 'Korea, South', 'St. Vincent & Grenadines': 'Saint Vincent and the Grenadines', 'Taiwan': 'Taiwan*', 'United States': 'US','State of Palestine': 'West Bank and Gaza', 'Côte d\'Ivoire': 'Cote d\'Ivoire'})
    list_new_country = [pd.Series(['Diamond Princess', 3711], index = reqd_pop_list.columns ) ,
                    pd.Series(['Kosovo', 1845000], index = reqd_pop_list.columns ) ,
                    pd.Series(['MS Zaandam', 1432], index = reqd_pop_list.columns ),
                    pd.Series(['Saint Kitts and Nevis', 52441], index = reqd_pop_list.columns ),
                    pd.Series(['Sao Tome and Principe', 211028], index = reqd_pop_list.columns )]
    reqd_pop_list = reqd_pop_list.append(list_new_country, ignore_index=True)\
                                .sort_values('country')\
                                .reset_index(drop=True)
    reqd_pop_list.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/raw/population_data.csv',sep=';',index=False)
    print('Number of rows in population dataframe: '+str(reqd_pop_list.shape[0]))


if __name__ == '__main__':
    get_johns_hopkins()
    get_current_data_germany()
    get_population_data()
