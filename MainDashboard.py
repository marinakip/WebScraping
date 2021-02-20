import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import TextProssesing2
import Clustering


# bootstrap =  "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/solar/bootstrap.min.css"
# app = dash.Dash(external_stylesheets=[bootstrap])


categories = ['Followers', 'Following', 'Stars', 'Contributions']
continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
programming_languages = ['Java', 'Python', 'R', 'JavaScript', 'Kotlin', 'Jupyter Notebook', 'TSQL', 'TypeScript',
                            'MATLAB', 'Ruby', 'Objective C', 'C#', 'HTML']

app = dash.Dash(__name__, title = 'GitHub\'s Users Location', external_stylesheets = [dbc.themes.SOLAR])

app.layout = html.Div([
    html.H1(" Locations of GitHub Users", style = {'text-align': 'center'}),
    html.Div([
        html.Div(dcc.Input(id = 'input-box', type = 'text', className = 'form-control mr-sm-2',
                    placeholder = 'Search for repositories',  debounce=True), style ={'width': '80%',
                                                                      'display': 'inline-block'}),
        html.Div(
            html.Button(children ='Search', id = 'search-button', n_clicks = 0,
                                className = 'btn btn-secondary my-2 my-sm-0'),
                                style = {'display': 'inline-block', 'width': '18%', 'float': 'center',
                                         'padding-left': '4px', 'padding-bottom': '2px'}
        ),

        #html.Div(id = 'test', style = {'display': 'inline-block', 'width': '40%', 'font': '16px'}),
        html.Div(id = 'test'),

    ]),  #DIV SEARCH

    html.Div([
        html.Div([
            html.Div([
                html.Label('Cluster by Category'),
                dcc.Dropdown(
                    id = 'filter-category',
                    options = [{'label': i, 'value': i} for i in categories],
                    placeholder = 'Select..',
                    clearable = False)

            ]), #DIV DROPDOWN CATEGORY

        ], style = {'width': '45%', 'font-size': '20px', 'float': 'right', 'display': 'inline-block'}), #DIV CATEGORY

        html.Div([
            html.Label('Cluster by Language'),
            dcc.Dropdown(
                id = 'filter-language',
                options = [{'label': i, 'value': i} for i in programming_languages],
                placeholder = 'Select..',
                clearable = False
            )
        ], style = {'width': '45%', 'font-size': '20px', 'display': 'inline-block'}), #DIV CONTINENT

        html.Div([])


    ]), #DIV FILTERS
    html.Br(),
    html.Br(),
    html.Div([
        dbc.Spinner(dcc.Graph(id = 'world-map')),

        html.Br(),
        # html.Br(),
        #
        # dcc.Graph(
        #     id = 'cluster-plot',
        #     hoverData = {}
        # )

    ], style = {'width': '100%', 'height': '90%', 'display': 'inline-block', 'padding': '0 20'}) #DIV GRAPHS


])


@app.callback(
        dash.dependencies.Output('world-map', 'figure'),
        # [dash.dependencies.Output('world-map', 'figure'),
        #  dash.dependencies.Output('cluster-plot', 'figure')],
        [
            dash.dependencies.Input('input-box', 'value'),
            dash.dependencies.Input('search-button', 'n_clicks'),
            dash.dependencies.Input('filter-category', 'value'),
            dash.dependencies.State('input-box', 'value'),
            dash.dependencies.State('filter-category', 'value')

        ]
)


def update_graph(input_value, clicks, category, input_state, category_state):
    #print("Category state: " + category_state)
    #print("Category: " + category)
    #print(input_value)  #clicks 0, input value  None
    if not clicks and not category:
        raise dash.exceptions.PreventUpdate
    if clicks > 0 or input_state is not None and category_state is not None:
        print("clicks -- input value -- input state")
        print(clicks, input_value, input_state)
        print("search bar Text: {}".format(input_value))
        print("Searching...")
        # figure1, df = TextProssesing2.process_query(input_value)
        df = TextProssesing2.process_query(input_value)
        #print("figure 1 ok, df and category")
        print(df.head(10))
        print(category)
        print("start figure clustering")
        figure = Clustering.clustering_with_weight(df, category)
        return figure
        #print("figure 2 ok")
        #print("figure 1 update")
        #figure1.update_layout(margin = {"r": 0, "t": 0, "l": 0, "b": 0})
        #print("figure 2 update")
        #figure2.update_layout(margin = {"r": 0, "t": 0, "l": 0, "b": 0})
        #print("all figures updated")
        #return [figure1, figure2]
        #return figure2
        #print(df)



if __name__ == "__main__":
    app.run_server(debug = True)
