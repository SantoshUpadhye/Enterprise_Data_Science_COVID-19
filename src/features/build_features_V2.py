import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal

from scipy import optimize
from scipy import integrate


def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate
        Parameters:
        ----------
        in_array : pandas.series
        Returns:
        ----------
        Doubling rate: double
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope



def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)
        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result
        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           1)
    df_result[str(column+'_filtered')]=result
    return df_result


def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'
        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)



    return result



def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame
        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()

    #print('--+++ after group by apply')
    #print(pd_filtered_result[pd_filtered_result['country']=='Germany'].tail())

    #df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    #print(df_output[df_output['country']=='Germany'].tail())
    return df_output.copy()



def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame
        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])

    return df_output


def SIR_model_and_fitting_parameters():
    df_analyse = pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/all_country_flat_table.csv',sep=';',parse_dates=[0])

    population_df = pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/raw/population_data.csv',sep=';', thousands=',')
    population_df = population_df.set_index(['country']).T


    ydata = []

    for column in df_analyse.columns:
        ydata.append(np.array(df_analyse[column][75:]))

    ydata_df = pd.DataFrame(ydata,index=df_analyse.columns).T
    ydata_df.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/ydata_SIR_data.csv',sep=';',index=False)
    print('Number of rows in ydata_df: '+str(ydata_df.shape[0]))

    optimized_df = pd.DataFrame(columns = df_analyse.columns[1:],
                     index = ['opt_beta', 'opt_gamma', 'std_dev_error_beta', 'std_dev_error_gamma'])

    t = []
    fitted_final_data = []
    global I0, N0, S0, R0
    for column in ydata_df.columns[1:]:                        # 0th column is date, which should not be considered
        I0 = ydata_df[column].loc[0]                           # initialising I0 with first value of ydata_df dataframe for a country
        N0 = population_df[column].loc['population']           # initialising N0 with population of the country_list
        S0 = N0-I0                                             # calculating susceptible population = total population - infected population
        R0 = 0                                                 # initialising recovered to zero
        t  = np.arange(len(ydata_df[column]))                  # calculating number of days in arange format
        popt=[0.4,0.1]                                         # initialising beta and gamma

        fit_odeint(t, *popt)                                   # fitting SIR data to use it for getting optimize beta and gamma

        popt, pcov = optimize.curve_fit(fit_odeint, t, ydata_df[column], maxfev=5000)   # optimising beta and gamma
        perr = np.sqrt(np.diag(pcov))                   # variance and standard deviation

        optimized_df.at['opt_beta', column] = popt[0]
        optimized_df.at['opt_gamma', column] = popt[1]
        optimized_df.at['std_dev_error_beta', column] = perr[0]
        optimized_df.at['std_dev_error_gamma', column] = perr[1]

        fitted = fit_odeint(t, *popt)                   # calculating fitted curve for a country
        fitted_final_data.append(np.array(fitted))      # appending calculated values to a list


    fitted_SIR_data_df = pd.DataFrame(fitted_final_data,index=df_analyse.columns[1:]).T

    optimized_df.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/optimized_SIR_data.csv',sep=';',index=False)
    print('Number of rows in optimized dataframe: '+str(optimized_df.shape[0]))

    fitted_SIR_data_df.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/fitted_SIR_data.csv',sep=';',index=False)
    print('Number of rows in fitted_SIR_data: '+str(ydata_df.shape[0]))



def SIR_model_t(SIRN,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    '''

    S,I,R,N=SIRN
    dS_dt=-beta*S*I/N          #S*I is the
    dI_dt=beta*S*I/N-gamma*I
    dR_dt=gamma*I
    dN_dt=0
    return dS_dt,dI_dt,dR_dt,dN_dt


def fit_odeint(t, beta, gamma): #t==x?
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0, N0), t, args=(beta, gamma))[:,1] # we only would like to get dI


if __name__ == '__main__':
    test_data_reg=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data_reg)
    print('the test slope is: '+str(result))

    pd_JH_data=pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()

    #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
    #                  (pd_JH_data['country']=='Germany'))]

    pd_result_larg=calc_filtered_data(pd_JH_data)
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')


    mask=pd_result_larg['confirmed']>100
    pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_larg.to_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/COVID_final_set.csv',sep=';',index=False)
    print(pd_result_larg[pd_result_larg['country']=='Germany'].tail())

    SIR_model_and_fitting_parameters()
