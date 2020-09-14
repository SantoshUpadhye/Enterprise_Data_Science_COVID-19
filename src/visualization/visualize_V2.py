import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

import os
print(os.getcwd())
df_input_large=pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/COVID_final_set.csv',sep=';')

fitted_final_data_dash_df = pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/fitted_SIR_data.csv',sep=';')

optimized_dash_df = pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/optimized_SIR_data.csv',sep=';')

ydata_dash_df = pd.read_csv('C:/ProgramData/Anaconda3/eps_covid19/data/processed/SIR/ydata_SIR_data.csv',sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data
    Goal of the project is to learn data science by applying a cross industry standard process,
    it covers the full walkthrough of: automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard.

    * Extension of the existing model with SIR model was developed and incorporated in the same dashboard.
    '''),

    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),

    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany','Italy'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases, confirmed cases COVID-19 filtered, approximated doubling time, doubling time filtered or SIR curve with filtered SIR curve
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
        {'label': 'SIR curve and fitted SIR curve', 'value': 'SIR_value'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):


    if 'DR' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    elif 'SIR' in show_doubling:
        my_yaxis={'type': "log",
               'title':'Population infected'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling == 'SIR_value':
            traces.append(dict(x=ydata_dash_df.date, #t, #df_plot.date,
                                y=ydata_dash_df[each], #=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                line_width=2,
                                marker_size=4,
                                name=each

                        )
                )
            traces.append(dict(x=ydata_dash_df.date, #df_plot.date,
                                y=fitted_final_data_dash_df[each], #=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                line_width=2,
                                marker_size=4,
                                name=each+'_fitted'

                        )
                )


        elif 'DR' in show_doubling: #elif show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
            traces.append(dict(x=df_plot.date,
                                    y=df_plot[show_doubling],
                                    mode='markers+lines',
                                    opacity=0.9,
                                    name=each
                            )
                    )

        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)
            traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
            
    }


if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)
