#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash, dcc, html, Input, Output
from dash import jupyter_dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

jupyter_dash.default_mode="external"

# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [

    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = Dash(prevent_initial_callbacks="initial_duplicate",
           external_scripts=external_scripts,
           external_stylesheets=external_stylesheets,
           title="test_app")
            
server = app.server

# Path
script_dir = os.path.dirname(__file__)  # the cwd relative path of the script file
rel_path = "data/result.csv"  # the target file
rel_to_cwd_path = os.path.join(script_dir, rel_path)  # the cwd-relative path of the target file

# Read csv file
df = pd.read_csv(rel_to_cwd_path)

color_palette = ['#4C78A8', '#E45756', '#54A24B', '#72B7B2', '#F6912F',
                 '#64a6bd', '#fd8595', '#85d48b', '#f45f34', '#ffcc33'
                 ]
orientation_list = ['positive', 'negative', 'positive', 'negative', 'positive']


target_list = ['Final_G']
con_features = ['absences', 'Final_G']
cat_features = ['school', 
                'sex', 
                'age', 
                'address', 
                'famsize', 
                'Pstatus',
                'Medu', 
                'Fedu', 
                'Mjob', 
                'Fjob', 
                'reason', 
                'guardian', 
                'traveltime',
                'studytime', 
                'failures', 
                'schoolsup', 
                'famsup', 
                'paid', 
                'activities',
                'nursery', 
                'higher', 
                'internet', 
                'romantic', 
                'famrel', 
                'freetime',
                'goout', 
                'Dalc', 
                'Walc', 
                'health', 
                'Subject',
                ]

colors = {
    'background': 'rgba(66,154,234, 0.1)',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background'],  'padding': '25px 25px 75px 25px'}, children=[
    html.H1("Student performance dashboard", style={'text-align': 'center', 'color': '#4C78A8'}),
    # Target section
    html.Div([
        html.Div([
            html.H5('Select target column  ', style={'display': 'inline-block',
                                                     'margin-right': 5,
                                                     'vertical-align': 'middle'}
                    ),
            dcc.Dropdown(
                options=target_list,
                value='Final_G',
                id='dropdown_target',
                style={'width': '150px',
                       'display': 'inline-block',
                       'vertical-align': 'middle',
                       },
                clearable=False,
                )],
            className="four columns",
            style={'margin-left': '10px'},),
        ],
        className="row",
    ),
    
    #  Categorical feature
    html.Div([
        html.Div([
            html.H3('Univariate analysis'),
            html.H6('Select categorical column  ',
                    style={'display': 'inline-block',
                           'margin-right': 5,
                           'vertical-align': 'middle'}),
            dcc.Dropdown(
                options=cat_features,
                value='Fjob',
                id='dropdown_cat',
                style={'width': '150px',
                       'display': 'inline-block',
                       'vertical-align': 'middle',
                       },
                clearable=False,
            ),            
        ],
            className="four columns columns",
            style={'margin-left': '10px'},
        ),

    # Categorical Bivariate analysis
        html.Div([
            html.H3('Bivariate analysis'),
            dcc.RadioItems(
                id='radio_cat', 
                options=['box', 'violin'],
                value='box', 
                style={
                   'display':'inline-block', 
                   'vertical-align': 'middle',
                  },
                inline=True
            ),
            html.H6('Add separation ',
                    style={'display': 'inline-block',
                           'margin-left': 25,
                           'margin-right': 5,
                           'vertical-align': 'middle'}),
            dcc.Dropdown(
                options=['None'] + cat_features,
                value='None',
                id='dropdown_cat_add',
                style={'width': '150px',
                       'display': 'inline-block',
                       'vertical-align': 'middle',},
                clearable=False,
                ),
        ], 
            className="eight columns",
            style={'margin-left': '-10px'},),
    ], 
    className="row"),
    
    ### charts section 
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='chart1'),
            ],
            className="eight columns columns",
            style={'display': 'inline-block'},
            ),
            html.Div([
                html.Div(id='cat_stats', style={'display': 'inline-block',
                                                'whiteSpace': 'pre-line',
                                                'vertical': ':middle',
                                                'font-size': '1.2rem',
                                                'line-height': '90%',
                                                'font-family': "Lucida Console"
                                                }
                         ),
            ],
            className="four columns columns",
            style={'display': 'inline-block'},
            ),
        ],
        className="four columns columns",
        style={'display': 'inline-block', 'margin-left': '10px'},
        ),
        
        html.Div([
            dcc.Graph(id='chart2'),
        ],
            className="eight columns",
            style={'display': 'inline-block', 'margin-left': '-10px'},),
    ],
        className="row"),
    
    # Continuous feature

    html.Div([
        html.Div([
            html.H6('Select continuous column  ',
                    style={'display': 'inline-block',
                           'margin-right': 5,
                           'vertical-align': 'middle'}),
            dcc.Dropdown(
                options=con_features,
                value='Final_G',
                id='dropdown_con',
                style={'width': '150px',
                       'display': 'inline-block',
                       'vertical-align': 'middle',
                       },
                clearable=False,
            ),            
        ], 
            className="four columns columns",
            style={'margin-left': '10px'},
        ),
    # Continuous Bivariate analysis
        html.Div([
            html.H6('Add separation ',
                    style={'display': 'inline-block',
                           'margin-left': 25,
                           'margin-right': 5,
                           'vertical-align': 'middle'}),
            dcc.Dropdown(
                options=['None'] + cat_features,
                value="None",
                id='dropdown_con_add',
                style={'width':'150px',
                       'display':'inline-block', 
                       'vertical-align': 'middle',},
                clearable=False,
                ),
        ], 
            className="eight columns",
            style={'margin-left': '-10px'},),
    ],
        className="row"),
    
    # Charts section
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='chart3'),
            ],
            className="eight columns columns",
            style={'display': 'inline-block'},
            ),
            html.Div([
                html.Div(id='con_stats', style={'display': 'inline-block',
                                                'whiteSpace': 'pre-line',
                                                'vertical': ':middle',
                                                'font-size': '1.2rem',
                                                'line-height': '90%',
                                                'font-family': "Lucida Console"}
                         ),
            ],
                className="four columns columns",
                style={'display': 'inline-block'},
            ),
        ],
            className="four columns columns",
            style={'display': 'inline-block', 'margin-left': '10px'},
        ),
        
        html.Div([
            dcc.Graph(id='chart4'),
        ],
            className="eight columns",
            style={'display': 'inline-block', 'margin-left': '-10px'},),
    ],
        className="row"),
])


def create_box(df, feature, feature_sep, target, fig):
    #fig = go.Figure()
    if feature_sep == 'None':
        fig.add_trace(go.Box(x=df[feature], y=df.Final_G, marker_color='#4C78A8', boxmean='sd'))
    else:
        name_list = list(df[feature_sep].unique())
        for name, color in zip(sorted(name_list), color_palette):
            fig.add_trace(go.Box(x=df[df[feature_sep] == name][feature], 
                                 y=df[target],
                                 name=str(name),
                                 marker_color=color,
                                 boxmean='sd',
                                 marker=dict(
                                    outliercolor='rgba(219, 64, 82, 0.6)',
                                    line=dict(
                                        outliercolor='rgba(219, 64, 82, 0.6)',
                                        outlierwidth=2)),
                                   ))
        fig.update_layout(boxmode='group')
    return fig


def create_violin(df, feature, feature_sep, target, fig):
    if feature_sep=='None':
        fig.add_trace(go.Violin(x=df[feature], y=df[target], line_color='#4C78A8'))
    else:
        name_list = list(df[feature_sep].unique())
        for name, color, orientation in zip(sorted(name_list), color_palette, orientation_list):
            fig.add_trace(go.Violin(x=df[df[feature_sep] == name][feature], 
                                    y=df[target],
                                    name=str(name),
                                    side=orientation,
                                    line_color=color
                                   ))
            fig.update_layout(violingap=0.4, violinmode='overlay')
    fig.update_traces(meanline_visible=True) 
    return fig


def my_chart_bivar(chart_type, target, cat_feature, sep_feature):
    fig = go.Figure()
    if sep_feature != 'None':
        if chart_type == 'box':
            fig = create_box(df, cat_feature, sep_feature, target, fig)
        elif chart_type == 'violin':
            fig = create_violin(df, cat_feature, sep_feature, target, fig)   
    else:
        if chart_type == 'box':
            fig = create_box(df, cat_feature, 'None', target, fig)
        elif chart_type == 'violin':
            fig = create_violin(df, cat_feature, 'None', target, fig)   
            
    fig.update_layout(
            autosize=True,
            #width=1200,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="LightSteelBlue",
            yaxis_title=target
        )
    return fig


@app.callback(
    Output(component_id='chart1', component_property='figure', allow_duplicate=True),
    Input(component_id='dropdown_cat', component_property='value'),
)
def generate_bar_chart(feature):
    val_counts = df[feature].value_counts()
    hovertext = []
    for i in (df[feature].value_counts()/len(df)):
        hovertext.append(f'{i*100:.1f}%')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=val_counts.index,
                         y=val_counts.values, 
                         hovertext=hovertext,
                         marker_color='#4C78A8',
                        ))
    fig.update_layout(
        autosize=True,
        #width=400,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    return fig


@app.callback(
    Output(component_id='cat_stats', component_property='children', allow_duplicate=True),
    Input(component_id='dropdown_cat', component_property='value'),
    Input(component_id='dropdown_target', component_property='value')
)


def update_output(value, target):
    var_group = df.groupby(value)[target].var()
    stat_string = "Var for " + str(var_group).split('Name')[0]
    return stat_string


@app.callback(
    Output(component_id='con_stats', component_property='children', allow_duplicate=True),
    Input(component_id='dropdown_con', component_property='value'),
    Input(component_id='dropdown_target', component_property='value')
)

def update_output(value, target):
    var_group = df.groupby(value)[target].var()
    stat_string = "Statistics for " + str(value)+ '\n' + str(df[value].describe()).split('Name')[0]
    return stat_string


# In[13]:


@app.callback(
    Output(component_id='chart2', component_property='figure', allow_duplicate=True),
    Input(component_id='radio_cat', component_property='value'),
    Input(component_id='dropdown_target', component_property='value'),
    Input(component_id='dropdown_cat', component_property='value'),
    Input(component_id='dropdown_cat_add', component_property='value'),
)
def generate_chart(chart_type, target, cat_feature, sep_feature):
    return my_chart_bivar(chart_type, target, cat_feature, sep_feature) 


# In[14]:


def create_hist(feature, sep_feature, fig):
    name_list = df[sep_feature].unique()
    for name_f, color in zip(name_list, color_palette):
        fig.add_trace(go.Histogram(x=df[df[sep_feature] == name_f][feature], marker_color=color, histnorm='probability')
                      )
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.4)
    return fig


@app.callback(
    Output(component_id='chart3', component_property='figure', allow_duplicate=True),
    Input(component_id='dropdown_con', component_property='value'))
def generate_chart(feature):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[feature], marker_color=color_palette[0]))
    fig.update_layout(
        autosize=True,
        #width=400,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    return fig


@app.callback(
    Output(component_id='chart4', component_property='figure', allow_duplicate=True),
    Input(component_id='dropdown_con', component_property='value'),
    Input(component_id='dropdown_con_add', component_property='value'))
def generate_chart(feature, sep_feature):
    fig = go.Figure()
    if sep_feature != 'None':
        fig = create_hist(feature, sep_feature, fig)
    else:
        fig.add_trace(go.Histogram(x=df[feature], marker_color=color_palette[0], histnorm='probability'))

    fig.update_traces(opacity=0.4)
    fig.update_layout(
        autosize=True,
        # width=400,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    return fig



if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)



