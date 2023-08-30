import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings("ignore")

import requests

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


# Create empty lists to store weak against and effective against types
weak_against = []
effective_against = []

attrib = ['against_normal',
             'against_fire',
             'against_water',
             'against_electric',
             'against_grass',
             'against_ice',
             'against_fight',
             'against_poison',
             'against_ground',
             'against_flying',
             'against_psychic',
             'against_bug',
             'against_rock',
             'against_ghost',
             'against_dragon',
             'against_dark',
             'against_steel',
             'against_fairy']

# Iterate through each row in the DataFrame
for index, row in df_poke.iterrows():
    # Reset lists for each Pokemon
    weak_against = []
    effective_against = []
    
    # Iterate through the attributes
    for att in attrib:
        # Get the value of the current attribute
        value = row[att]
        
        # Check if the Pokemon is weak or effective against this type
        if value < 1:
            effective_against.append(att[8:].replace('_', ' ').capitalize())  # Removing 'against_' and formatting
        elif value > 1:
            weak_against.append(att[8:].replace('_', ' ').capitalize())  # Removing 'against_' and formatting
    
    # Update the DataFrame with the lists of weak and effective against types
    df_poke.at[index, 'Effective_against'] = ', '.join(effective_against)
    df_poke.at[index, 'Weak_against'] = ', '.join(weak_against)


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

excluded_words = ['Mega', 'Alolan', 'Partner', 'Galarian']
pokemon_names = [name for name in pokemon_names if not any(word in name for word in excluded_words)]

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

        # Check if pokemon1 is effective against pokemon2
    is_pokemon1_effective = any(type_ in pokemon1_data['Effective_against'] for type_ in [pokemon2_data['type_1'], pokemon2_data['type_2']])
    
    # Check if pokemon2 is effective against pokemon1
    is_pokemon2_effective = any(type_ in pokemon2_data['Effective_against'] for type_ in [pokemon1_data['type_1'], pokemon1_data['type_2']])
    

    # Determine the winner
    if is_pokemon1_effective and not is_pokemon2_effective:
        winner = pokemon1
    elif is_pokemon2_effective and not is_pokemon1_effective:
        winner = pokemon2
    else:
        winner = "No one"  # If both are effective against each other

        
        # Create div elements for displaying the winner
    winner_div = html.Div([
        html.H2("Winner:", style={'color': 'green'}),
        html.H1(winner, style={'font-size': '40px'}),
    ], className='winner-info')
    
    
    # Create div elements for displaying Pokémon details
    pokemon1_details = html.Div([
        html.H2(pokemon1),
        html.Img(src=f"data:image/png;base64,{read_image_from_zip(zip_url, f'{pokemon1}.png')}"),
        html.Table([
            html.Tr([html.Th("Type 1"), html.Td(pokemon1_data['type_1'])]),
            html.Tr([html.Th("Type 2"), html.Td(pokemon1_data['type_2'])]),
            html.Tr([html.Th("Effective against"), html.Td(pokemon1_data['Effective_against'])]),
            html.Tr([html.Th("Weak against"), html.Td(pokemon1_data['Weak_against'])]),
        ]),
    ], className='pokemon-info')

    pokemon2_details = html.Div([
        html.H2(pokemon2),
        html.Img(src=f"data:image/png;base64,{read_image_from_zip(zip_url, f'{pokemon2}.png')}"),
        html.Table([
            html.Tr([html.Th("Type 1"), html.Td(pokemon2_data['type_1'])]),
            html.Tr([html.Th("Type 2"), html.Td(pokemon2_data['type_2'])]),
            html.Tr([html.Th("Effective against"), html.Td(pokemon2_data['Effective_against'])]),
            html.Tr([html.Th("Weak against"), html.Td(pokemon2_data['Weak_against'])]),
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
            winner_div,
        ], className='pokemon-details'),

        bar_plot_div,
    ], className='pokemon-container')

    return info_div

# ... (remaining parts of the code)



if __name__ == '__main__':
    app.run_server(debug=True)




if __name__ == '__main__':
    app.run_server(debug=True)
