#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings("ignore")



import dash

import dash_core_components as dcc

import dash_html_components as html

from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px



# URL of the raw CSV file on GitHub

csv_url = "https://raw.githubusercontent.com/manansatsangi/Pokemon/master/pokedex_(Update_05.20).csv"



# Read the CSV file into a pandas DataFrame

df_pokemon = pd.read_csv(csv_url)


df_poke=df_pokemon.copy()
df_poke['egg_type_2'].fillna('None', inplace = True)
df_poke['ability_2'].fillna('None', inplace = True)
df_poke['type_2'].fillna('None', inplace = True)
df_poke['percentage_male'].fillna(0, inplace = True)
df_poke['base_friendship'].fillna(0, inplace = True)
df_poke['base_experience'].fillna(0, inplace = True)
df_poke['catch_rate'].fillna(0, inplace = True)
df_poke['japanese_name'].fillna('Unknown', inplace = True)
df_poke['german_name'].fillna('Unknown', inplace = True)
df_poke['egg_type_1'].fillna('Unknown', inplace = True)
df_poke['ability_1'].fillna('Unknown', inplace = True)
df_poke['growth_rate'].fillna('Unknown', inplace = True)
df_poke['weight_kg'].fillna(0, inplace = True)
df_poke['egg_cycles'].fillna(0, inplace = True)


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import zipfile
from io import BytesIO
import base64
import plotly.graph_objs as go

# Create a Dash app
app = dash.Dash(__name__)
server = app.server

# Load the Pokemon data
df_pokemon = df_poke.copy()  # Replace with your Pokemon data

# Load the list of Pokemon names
pokemon_names = df_pokemon['name'].tolist()

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Pokemon Versus Mode"),

    # Dropdowns to select two Pokemon
    dcc.Dropdown(id='pokemon1', options=[{'label': name, 'value': name} for name in pokemon_names], value=pokemon_names[0]),
    dcc.Dropdown(id='pokemon2', options=[{'label': name, 'value': name} for name in pokemon_names], value=pokemon_names[1]),

    # Display Pokemon attributes and images
    html.Div(id='pokemon-info'),
])

# Function to read and encode an image from a zip file
zip_url = "https://github.com/manansatsangi/Pokemon/raw/main/pokemon_images2.zip"

def read_image_from_zip(zip_url, image_name):
    response = requests.get(zip_url)
    zip_data = BytesIO(response.content)
    
    with zipfile.ZipFile(zip_data) as zf:
        image_data = zf.read(image_name)
    
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return encoded_image


# ... (other import statements and app setup)

# Callback to update the displayed information
@app.callback(
    Output('pokemon-info', 'children'),
    [Input('pokemon1', 'value'),
     Input('pokemon2', 'value')]
)
def update_pokemon_info(pokemon1, pokemon2):
    # Find the data for the selected Pokemon
    pokemon1_data = df_pokemon[df_pokemon['name'] == pokemon1].iloc[0]
    pokemon2_data = df_pokemon[df_pokemon['name'] == pokemon2].iloc[0]

    # Create div elements for displaying Pokémon details
    pokemon1_details = html.Div([
        html.H2(pokemon1),
        html.Img(src=f"data:image/png;base64,{read_image_from_zip(zip_url, f'{pokemon1}.png')}"),
        html.Table([
            html.Tr([html.Th("Type 1"), html.Td(pokemon1_data['type_1'])]),
            html.Tr([html.Th("Type 2"), html.Td(pokemon1_data['type_2'])]),
        ]),
    ], className='pokemon-info')

    pokemon2_details = html.Div([
        html.H2(pokemon2),
        html.Img(src=f"data:image/png;base64,{read_image_from_zip(zip_url, f'{pokemon2}.png')}"),
        html.Table([
            html.Tr([html.Th("Type 1"), html.Td(pokemon2_data['type_1'])]),
            html.Tr([html.Th("Type 2"), html.Td(pokemon2_data['type_2'])]),
        ]),
    ], className='pokemon-info')

    # Create a grouped horizontal bar plot
    attributes = ['attack', 'sp_attack', 'defense', 'sp_defense', 'speed']
    y_values = attributes
    x_values1 = [pokemon1_data[attr] for attr in attributes]
    x_values2 = [-pokemon2_data[attr] for attr in attributes]  # Negative to show on the left side

    bar_plot1 = go.Bar(y=y_values, x=x_values1, orientation='h', name=pokemon1)
    bar_plot2 = go.Bar(y=y_values, x=x_values2, orientation='h', name=pokemon2)

    bar_layout = go.Layout(barmode='relative', yaxis={'categoryorder': 'total ascending'})
    bar_fig = go.Figure(data=[bar_plot1, bar_plot2], layout=bar_layout)
    bar_plot_div = dcc.Graph(figure=bar_fig)

    # Create div elements for displaying information
    info_div = html.Div([
        html.Div([
            pokemon1_details,
            pokemon2_details,
        ], className='pokemon-details'),

        bar_plot_div,
    ], className='pokemon-container')

    return info_div

# ... (remaining parts of the code)



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


# In[ ]:




