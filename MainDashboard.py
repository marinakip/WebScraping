import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import mysql.connector
import pandas.io.sql as psql

# db = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="marina1992",
#   database="locations"
# )
#
# print(db)
# query = "SELECT * FROM addresses_geocoded WHERE weight > 2000"
#=================================================================
# cursor = db.cursor()
# cursor.execute(query)
# addresses = cursor.fetchmany(10)
#print("Total number of rows is: ", cursor.rowcount)

# for x in addresses:
#     print(x)
#===================================================
# df = psql.read_sql(query, con=db)
# print(df.head(10))

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1(" Locations of GitHub Users", style={'text-align': 'center'}),

    dcc.Dropdown(id="keywords",
                 options=[
                     {"label": "Diabetes", "value": "diabetes"},
                     {"label": "Machine Learning", "value": "machine learning"},
                     {"label": "Medical", "value": "medical"}],
                 multi=True,
                 value="diabetes",
                 style={'width': "70%"}
                 ),
    html.Br(),

    dcc.Slider(
        min=2016,
        max=2020,
        marks={i: '{}'.format(i) for i in range(2016, 2021)},
        value=2016,
    ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='map', figure={})

])

if __name__ == '__main__':
    app.run_server(debug=True)

# db.close()