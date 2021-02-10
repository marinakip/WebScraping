import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np

# bootstrap =  "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/solar/bootstrap.min.css"
# app = dash.Dash(external_stylesheets=[bootstrap])

categories = ['Followers', 'Following', 'Stars', 'Contributions']
continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

app = dash.Dash(__name__, title = 'GitHub\'s Users Location', external_stylesheets = [dbc.themes.SOLAR])

app.layout = html.Div([
    html.H1(" Locations of GitHub Users", style = {'text-align': 'center'}),
    html.Div([
        html.Div(dcc.Input(id = 'input-box', type = 'text', className = 'form-control mr-sm-2',
                    placeholder = 'Search for repositories'), style ={'width': '80%',
                                                                      'display': 'inline-block'}),
        html.Div(
            html.Button('Search', id = 'button',  className = 'btn btn-secondary my-2 my-sm-0'),
                                style = {'display': 'inline-block', 'width': '18%', 'float': 'center',
                                         'padding-left': '4px', 'padding-bottom': '2px'}
        )

    ]),  #DIV SEARCH

    html.Div([
        html.Div([
            html.Div([
                html.Label('Cluster by Category'),
                dcc.Dropdown(
                    id = 'filter-category',
                    options = [{'label': i, 'value': i} for i in categories],
                    value = 'Followers',
                    clearable = False)

            ]), #DIV DROPDOWN CATEGORY

        ], style = {'width': '45%', 'font-size': '20px', 'float': 'right', 'display': 'inline-block'}), #DIV CATEGORY

        html.Div([
            html.Label('Cluster by Continent'),
            dcc.Dropdown(
                id = 'filter-continent',
                options = [{'label': i, 'value': i} for i in continents],
                value = 'Europe',
                clearable = False
            )
        ], style = {'width': '45%', 'font-size': '20px', 'display': 'inline-block'}), #DIV CONTINENT

        html.Div([])


    ]), #DIV FILTERS
    html.Br(),
    html.Div([
        dcc.Graph(
            id = 'world-map',
            hoverData = {}
        ),

        html.Br(),
        html.Br(),

        dcc.Graph(
            id = 'scatter-plot',
            hoverData = {}
        )

    ], style = {'width': '100%', 'height': '90%', 'display': 'inline-block', 'padding': '0 20'}) #DIV GRAPHS
    @app.callback(
        dash.dependencies.Output('world-map', 'figure'),
        [
            dash.dependencies.Input('input-box', 'value'),
            dash.dependencies.Input('filter-category', 'value'),
            dash.dependencies.Input('filter-continent', 'value')
        ]
    )

])



if __name__ == "__main__":
    app.run_server(debug = True)
