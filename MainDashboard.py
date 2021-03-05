import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import numpy as np
import TextProssesing2
import Clustering

# bootstrap =  "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/solar/bootstrap.min.css"
# app = dash.Dash(external_stylesheets=[bootstrap])

cluster_by = ['Feature', 'Weighted Feature', 'Auto Clustering']
categories = ['Followers', 'Following', 'Stars', 'Contributions']
programming_languages = ['Java', 'Python', 'R', 'JavaScript', 'Kotlin', 'Jupyter Notebook', 'TSQL', 'TypeScript',
                         'MATLAB', 'Ruby', 'Objective C', 'C#', 'HTML']

app = dash.Dash(__name__, title = 'GitHub\'s Users Location', external_stylesheets = [dbc.themes.SOLAR])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.H1(" Locations of GitHub Users", style = {'text-align': 'center'}),
    html.Div([
        html.Div(dcc.Input(id = 'input-box', type = 'text', className = 'form-control mr-sm-2',
                           placeholder = 'Search for repositories', debounce = True), style = {'width': '80%',
                                                                                               'display': 'inline-block'}),
        html.Div(
            html.Button(children = 'Search', id = 'search-button', n_clicks = 0,
                        className = 'btn btn-secondary my-2 my-sm-0'),
            style = {'display': 'inline-block', 'width': '18%', 'float': 'center',
                     'padding-left': '4px', 'padding-bottom': '2px'}
        ),

        # html.Div(id = 'test', style = {'display': 'inline-block', 'width': '40%', 'font': '16px'}),
        html.Div(id = 'test'),

    ]),  # DIV SEARCH

    html.Div([
        html.Div([
            html.Div(id = 'filter-container', children = [
                html.Label('Cluster by'),
                dcc.Dropdown(
                    id = 'filter-clustering',
                    options = [{'label': i, 'value': i} for i in cluster_by],
                    placeholder = 'Select..',
                    clearable = True)

            ]),  # DIV DROPDOWN CLUSTERING

        ], style = {'width': '45%', 'font-size': '20px', 'float': 'left', 'display': 'inline-block'}),  # DIV CATEGORY

        html.Div(id = 'feature-container', children = [
            html.Label('Select Feature'),
            dcc.Dropdown(
                id = 'filter-categories',
                options = [],
                placeholder = 'Select..',
                clearable = True
            )
        ], style = {'width': 'auto', 'font-size': '20px', 'float': 'right', 'display': 'none'}),  # DIV FEATURE

        html.Br(),
        html.Div(id = 'input-weights', children = [
            html.Br(),
            dcc.Input(id = "followers_weight", type = "number", placeholder = "Followers Weight"),
            dcc.Input(id = "following_weight", type = "number", placeholder = "Following Weight"),
            dcc.Input(id = "stars_weight", type = "number", placeholder = "Stars Weight"),
            dcc.Input(id = "contributions_weight", type = "number", placeholder = "Contributions Weight"),
            # dcc.Input(id="input2", type="text", placeholder="", debounce=True),
            html.Div(id = "output"),
        ], style = {'width': 'auto', 'font-size': '20px', 'float': 'right', 'display': 'none'})

    ]),  # DIV FILTERS
    html.Br(),
    html.Br(),
    html.Div([

        html.Div(id = 'hidden-div', children = [], style = {'display': 'none'}),
        html.Br(),
        html.Div(id = 'graph-container', children = [
            dbc.Spinner(dcc.Graph(id = 'world-map'))]),

        html.Br(),
        html.Br(),
        html.Div(dash_table.DataTable(id = "table",
                                      columns = [],
                                      data = [],
                                      style_header = {'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                      style_data_conditional = [{
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(248, 248, 248)'
                                      }]
                                      )

                 ),

    ], style = {'width': '100%', 'height': '90%', 'display': 'inline-block', 'padding': '0 20'})  # DIV GRAPHS

])


@app.callback(
    dash.dependencies.Output('feature-container', 'style'),
    dash.dependencies.Output('filter-categories', 'options'),
    dash.dependencies.Output('input-weights', 'style'),
    # dash.dependencies.Output('graph-container', 'style'),
    dash.dependencies.Input('filter-clustering', 'value')
)
def style_categories(clustering_technique):
    if not clustering_technique:
        raise dash.exceptions.PreventUpdate
    elif clustering_technique == 'Feature':
        print("selected: " + str(clustering_technique))
        print("Mpike")
        style_categories = {'width': '65%', 'font-size': '20px', 'float': 'center', 'display': 'inline-block'}
        style_weights = {'width': '65%', 'font-size': '20px', 'float': 'center', 'display': 'none'}
        categories_list = [{'label': i, 'value': i} for i in categories]
        # style_graph = {'display': 'block'}
        # print(categories_list)
        return style_categories, categories_list, style_weights
    elif clustering_technique == 'Weighted Feature':
        print("WEIGHTED_FEATURE")
        style_categories = {'width': '65%', 'font-size': '20px', 'float': 'center', 'display': 'none'}
        style_weights = {'width': '65%', 'font-size': '20px', 'float': 'center', 'display': 'inline-block'}
        # style_graph = {'display': 'block'}
        categories_list = []
        return style_categories, categories_list, style_weights
    else:

        return [None]


@app.callback(
    # dash.dependencies.Output('graph-container', 'style'),
    dash.dependencies.Output('table', 'columns'),
    dash.dependencies.Output('table', 'data'),
    # dash.dependencies.Output('table', 'style_cell_conditional'),
    # dash.dependencies.Output('table', 'style_data_conditional'),
    # dash.dependencies.Output('table', 'style_header'),
    dash.dependencies.Output('world-map', 'figure'),

    # dash.dependencies.Output('hidden-div', 'children'),
    # [dash.dependencies.Output('world-map', 'figure'),
    #  dash.dependencies.Output('cluster-plot', 'figure')],
    [
        dash.dependencies.Input('input-box', 'value'),
        dash.dependencies.Input('search-button', 'n_clicks'),
        # dash.dependencies.Input('filter-clustering', 'value'),
        dash.dependencies.Input('filter-categories', 'value'),
        dash.dependencies.State('input-box', 'value'),
        # dash.dependencies.State('filter-clustering', 'value')
        dash.dependencies.State('filter-categories', 'value'),

    ]
)
def update_graph(input_value, clicks, category, input_state, category_state):
    # print("Category state: " + category_state)
    # print("Category: " + category)
    # print(input_value)  #clicks 0, input value  None
    if not clicks and not category:
        raise dash.exceptions.PreventUpdate
    if clicks > 0 or input_state is not None and category_state is not None:
        # print("clicks -- input value -- input state")
        # print(clicks, input_value, input_state)
        # print("search bar Text: {}".format(input_value))
        print("Searching...")
        # figure1, df = TextProssesing2.process_query(input_value)
        df = TextProssesing2.process_query(input_value)
        # print("figure 1 ok, df and category")
        # print(df.head(10))
        # print(category)
        print("start figure clustering")
        figure, df_cluster_elements_auto = Clustering.clustering_auto(df)
        columns = [{'name': i, 'id': i} for i in df_cluster_elements_auto.columns]
        data = df_cluster_elements_auto.to_dict('records')

        # figure = Clustering.clustering_with_weight(df, category)
        # style_graph = {'display': 'inline-block'}
        return columns, data, figure


#
# @app.callback(
#     dash.dependencies.Output('table', 'columns'),
#     dash.dependencies.Output('table', 'data'),
#     dash.dependencies.Input('hidden-div', 'children'),
# )
#
# def update_table(df_table):
#     if df_table.empty:
#         raise dash.exceptions.PreventUpdate
#     print("inside table")
#     print(df_table.head(10))
#     columns = [{'name': i, 'id': i} for i in df_table.columns]
#     data = df_table.to_dict('records')
#     return columns, data
#
#
#


if __name__ == "__main__":
    app.run_server(debug = True)
