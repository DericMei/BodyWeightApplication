####################################
#### Import necessary libraries ####
####################################
import dash
from dash import html, dcc, Input, Output, State #, dash_table
import plotly.express as px
import pandas as pd
#import numpy as np
import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from prophet import Prophet
import plotly.graph_objs as go
import pickle
#import torch.nn as nn
#import joblib
#import torch


########################
#### Database Setup ####
########################
# This part is to load local environmental variables for testing the application
#load_dotenv()
#database_url = os.getenv('DATABASE_URL')

# This Part is for deployment purposes to connect the database
database_url = os.environ.get('DATABASE_URL')
# Check if the URL starts with "postgres://" this is a weird error with postgresql on heroku
if database_url.startswith('postgres://'):
    # Replace it with "postgresql://"
    database_url = 'postgresql://' + database_url[len('postgres://'):]

# Connect to database
engine = create_engine(database_url)


###################
#### Functions ####
###################
# Function to load the prophet model from database
def load_model():
    with engine.connect() as conn:
        fetch_query = text("SELECT model_data FROM models ORDER BY training_date DESC LIMIT 1")
        result = conn.execute(fetch_query)
        model_data = result.fetchone()[0]
        model = pickle.loads(model_data)
        return model

'''
# Due to the fact that torch is too big, I cant deploy this to heroku
# Redefine the same LSTMModel
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        # Define the LSTM architecture
        self.lstm = nn.LSTM(input_dim, hidden_dims, num_layers=num_layers,
                            dropout=dropout_prob, batch_first=True)
        self.output_layer = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        # Pass the input through the LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last hidden state for the last time step
        last_time_step = lstm_out[:, -1, :]
        # Pass the last hidden state of the last time step through the output layer
        x = self.output_layer(last_time_step)
        return x

# Function to predict future weights using lstm model
def predict_future_weights_with_dates(df, model, scaler, sequence_length=7, future_days=7):
    # Normalize the input data
    weights = df['weight'].values.reshape(-1, 1)
    normalized_weights = scaler.transform(weights)

    # Create the last sequence from the latest data
    last_sequence = normalized_weights[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))

    # Convert to tensor
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

    # Prepare to collect predictions
    future_predictions = []
    future_dates = [df['date'].iloc[-1] + timedelta(days=i) for i in range(future_days)]

    # Predict future weights, starting from today
    for _ in range(future_days):
        with torch.no_grad():
            prediction = model(last_sequence).numpy()
        
        # Update the sequence with the new prediction
        new_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        last_sequence = torch.tensor(new_sequence, dtype=torch.float32)

        # Denormalize and store the prediction
        denormalized_prediction = scaler.inverse_transform(prediction).flatten()[0]
        future_predictions.append(denormalized_prediction)

    # Create a DataFrame with dates and predictions
    df_future_predictions = pd.DataFrame({
        'date': future_dates,
        'predicted_weight': future_predictions
    })
    return df_future_predictions

# Function to predict all existing datapoints
def predict_all_weights(df, model, scaler, sequence_length=7):
    # Normalize the input data
    weights = df['weight'].values.reshape(-1, 1)
    normalized_weights = scaler.transform(weights)
    
    # Create sequences for the entire dataset
    sequences = [normalized_weights[i:i + sequence_length] for i in range(len(normalized_weights) - sequence_length)]
    
    # Prepare the sequences for prediction
    sequences = np.array(sequences).reshape(-1, sequence_length, 1)
    sequence_tensor = torch.tensor(sequences, dtype=torch.float32)
    
    # Predict weights
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []
    with torch.no_grad():
        for seq in sequence_tensor:
            prediction = model(seq.unsqueeze(0)).numpy()  # Unsqueeze to add batch dimension
            prediction = scaler.inverse_transform(prediction).flatten()[0]
            predictions.append(prediction)

    # Prepare the DataFrame with actual and predicted weights
    prediction_dates = df['date'].iloc[sequence_length:]
    df_predictions = pd.DataFrame({
        'date': prediction_dates,
        'actual_weight': df['weight'].iloc[sequence_length:],
        'predicted_weight': predictions
    })
    return df_predictions

############################
#### LSTM Model Loading ####
############################
# Load the model
model_lstm = LSTMModel(1, 32, num_layers=1, dropout_prob=0)
model_lstm.load_state_dict(torch.load('model_lstm.pth'))
model_lstm.eval()

# Load the saved scaler
scaler = joblib.load('weight_scaler.pkl')

'''

#############################################################
#############################################################
################### The Dash App Layout #####################
#############################################################
#############################################################
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server variable for Heroku

# Define Dash app layout
app.layout = dbc.Container([

    # Interval component for triggering callbacks on the page load
    dcc.Interval(
    id='interval-component',
    interval=1*1000,  # in milliseconds
    n_intervals=0,
    max_intervals=1),

    # Spacer
    html.Div(style={'height': '7px'}),

    # Title Section of the application
    dbc.Card([
        dbc.CardHeader(html.H1(html.Strong("Eric Mei's BodyWeight Magic!"), className='text-center'), style={'background-color': '#000080'}),
        dbc.CardBody([
            # Basic Info
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H4(html.Strong("Weight Tracking Informations"), className='text-center')),
                        dbc.CardBody([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Table(
                                        [html.Tbody(html.Tr([
                                            html.Td(html.H4('Last Recorded Date:'), style={'padding-right': '30px'}), 
                                            html.Td(html.H4(id='last-recorded-date'), style={'color': '#FF7F50'}),
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.H4('Last Recorded Weight:'), style={'padding-right': '30px'}), 
                                            html.Td(html.H4(id='last-recorded-weight'), style={'color': '#FF7F50'})
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.H4(id='days-recording-text'), style={'padding-right': '30px'}), 
                                            html.Td(html.H4(id='days-recording'), style={'color': '#FF7F50'})
                                        ]))]
                                    , className= 'text-left')
                                ])
                            ])
                        ])
                    ]), width=4),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H4(html.Strong("Weekly Summary"), className='text-center')),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Table(
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Today's Date:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='week-status'), style={'color': '#FF7F50'}),
                                                ]))] +
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Last Week's Avg:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='past-week'), style={'color': '#FF7F50'})
                                                ]))] +
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Current Week's Avg:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='current-week'), style={'color': '#FF7F50'})
                                                ]))]
                                            , className= 'text-left')
                                        ])
                                    ])
                                ], width=5),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Table(
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Current Week Caloric Status:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='caloric-status'), style={'color': '#FF7F50'}),
                                                ]))] +
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Weekly Avg Net Change:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='weight-diff'), style={'color': '#FF7F50'})
                                                ]))] +
                                                [html.Tbody(html.Tr([
                                                    html.Td(html.H4("Estimated Daily Caloric Status:"), style={'padding-right': '20px'}), 
                                                    html.Td(html.H4(id='calorie-amount'), style={'color': '#FF7F50'})
                                                ]))]
                                            , className= 'text-left')
                                        ])
                                    ])
                                ], width=7),
                            ])
                        ])
                    ]), width=8)
            ]),

            # Spacer
            html.Div(style={'height': '7px'}),

            # Inputs section
            dbc.Row([
                # Passwords part
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H4("Prove You Are The Great Eric Mei!!", className='text-center')),
                        dbc.CardBody([
                            dbc.Row([
                                    dbc.Col(width=1),
                                    dbc.Col(
                                        dcc.Input(
                                            id='password-input',
                                            type='password',
                                            placeholder='Enter password to Record Body Weight',
                                            className='form-control',
                                            style={'width': '100%', 'display': 'inline-block', 'marginRight': '5px'}
                                        ), 
                                    width=6),
                                    dbc.Col(
                                        html.Button(
                                            'Unlock', 
                                            id='unlock-button', 
                                            className='btn-primary btn-lg',
                                            style={'width': '100%'}
                                        ),
                                    width=3)
                            ],className='align-items-center'),
                        ]), 
                    ]),
                width=6),

                # Weight Recording Parts
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("This Part is Reserved for The Great EricMei!", id='message_2_1', className='text-center')),
                        dbc.CardBody([
                            dbc.Row(html.Div(id='password-output')),
                            dbc.Row(html.H4("YOU!!💀 SHALL!!💀 NOT!!💀 PASS!!💀 YOU ARE NOT ERIC!!", id='message_2_2', className='text-center')),
                            dbc.Row(html.H4("Thank You for checking out my App! But only Eric can record weight~ 😊", id='message_2_3', className='text-center'))
                            ])
                    ], id='card-1'),
                    dbc.Card([
                        dbc.CardHeader(html.H4("Welcome Home! Lord Eric, Please Record Your Weight:", id='message_1', className='text-center')),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(
                                    dcc.Input(
                                        id='weight-input',
                                        type='number',
                                        min=70, max=95, step=0.1,
                                        className='form-control',
                                        style={'width': '100%', 'display': 'inline-block', 'marginRight': '5px'}
                                    ),
                                width=4, className='text-center'),
                                dbc.Col(
                                    html.Button(
                                        'Record Weight', 
                                        id='record-button', 
                                        className='btn-primary btn-lg',
                                        style={'width': '100%'}
                                    ),
                                width=3, className='text-center'),
                                dbc.Col(
                                    html.Div(id='weight-record-status'),
                                width=5, className='text-left')
                            ], className='align-items-center')
                        ])
                    ], id='card-2'),
                ], width=6, className='text-left')
            ]),
        ]),
    ]),

    # Spacer
    html.Div(style={'height': '7px'}),

    # Weight Data Visualization Section
    # Visualizations: Important Metrics Daily v.s. Weekly
    dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='weight-data-dropdown',
                        options=[
                            {'label': 'Daily 🆚 Weekly Avg', 'value': 'day_vs_week'},
                            {'label': 'All Recorded Data', 'value': 'all_available'},
                        ],
                        value='day_vs_week', className='text-left'
                    )    
                , width=2),
                dbc.Col(html.H2(html.Strong('Weight Data Plots'), className='text-center'), width=8, className='my-auto'),
                dbc.Col(
                    dbc.Button("Collapse ⬇️", id='collapse-button', className='custom-button', n_clicks=0),
                    width=2, className='text-end'
                )
            ], className='align-items-center')
        ], style={'background-color': '#000080'}),
        dbc.Collapse([
            dbc.CardBody(id='day-vs-week-body', children=[
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Daily Trends", className='text-center'),
                                dcc.Dropdown(
                                    id = 'dropdown-daily',
                                    options=[
                                        {'label': 'Current Week', 'value': 'current_week'},
                                        {'label': 'Last Week', 'value': 'last_week'},
                                        {'label': 'Current Month', 'value': 'current_month'},
                                        {'label': 'Past 6 Weeks', 'value': '6_weeks'},
                                        {'label': 'Past 3 Months', 'value': 'three_month'},
                                        {'label': 'Past 6 Months', 'value': 'six_month'},
                                        {'label': 'Past Year', 'value': 'one_year'},
                                    ],
                                    value='current_month' #default value
                                ),
                            ]),
                            dbc.CardBody([
                                dcc.Graph(id='daily', style={'padding': '0px', 'margin': '0px'})
                            ])
                        ]), 
                    width=6),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Weekly Trends", className='text-center'),
                                dcc.Dropdown(
                                    id = 'dropdown-weekly',
                                    options=[
                                        {'label': 'Current Month', 'value': 'current_month'},
                                        {'label': 'Past 6 Weeks', 'value': '6_weeks'},
                                        {'label': 'Past 3 Months', 'value': 'three_month'},
                                        {'label': 'Past 6 Months', 'value': 'six_month'},
                                        {'label': 'Past Year', 'value': 'one_year'},
                                    ],
                                    value='current_month' #default value
                                ),
                            ]),
                            dbc.CardBody([
                                dcc.Graph(id='weekly', style={'padding': '0px', 'margin': '0px'})
                            ])
                        ]), 
                    width=6),
                ]),
            ], style={'display': 'block'}),
            # Visualizations: Just for fun Overall Body weight trend
            dbc.CardBody(id='all-weight-data-body', children=[
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("All Body Weight Records Data Plot", className='text-center'),
                                dcc.Dropdown(
                                    id = 'dropdown-all',
                                    options=[
                                        {'label': 'Daily', 'value': 'daily'},
                                        {'label': 'Weekly Average', 'value': 'weekly'},
                                        {'label': 'Monthly Average', 'value': 'montly'},
                                        {'label': 'Seasonal Average', 'value': 'seasonaly'}
                                    ],
                                    value='daily' #default value
                                ),
                            ]),
                            dbc.CardBody([
                                dcc.Graph(id='all', style={'padding': '0px', 'margin': '0px'})
                            ])
                        ]), width=12)
                ])
            ], style={'display': 'none'})
        ],id='collapse-part')
    ]),

    # Spacer
    html.Div(style={'height': '7px'}),

    # Machine Learning Insights
    dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col('', width=2),
                dbc.Col(html.H2(html.Strong('Machine Learning Insights'), className='text-center'), width=8, className='my-auto'),
                dbc.Col('', width=2, className='text-end')
            ], className='align-items-center')
            ], style={'background-color': '#000080'}),
        dbc.CardBody(id='prophet-body',children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4('Forecasts, Insights, and Model Training', className='text-center'),
                        ]),
                        dbc.CardBody([
                            html.Table(
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("Day of Week:"), style={'padding-right': '30px'}), 
                                    html.Td(html.H4(id='day-of-week'), style={'color': '#FF7F50'})
                                ]))] +
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("Today's Body Weight Forecast:"), style={'padding-right': '30px'}), 
                                    html.Td(html.H4(id='today-forecast'), style={'color': '#FF7F50'}),
                                ]))] +
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("Recorded Weight for Today:"), style={'padding-right': '30px'}), 
                                    html.Td(html.H4(id='weight-today'), style={'color': '#FF7F50'})
                                ]))] +
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("Model Haven't Been Trained For:"), style={'padding-right': '30px'}), 
                                    html.Td(html.H4(id='model-train-time'), style={'color': '#FF7F50'})
                                ]))] +
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("Press Button To Train:"), style={'padding-right': '30px'}), 
                                    html.Td([
                                        html.H4('Only Eric Can Press! 😊', id='message_3'),
                                        html.Button('Train Model', id='button-train')
                                        ])
                                ]))] +
                                [html.Tbody(html.Tr([
                                    html.Td(html.H4("(Model Training Message:)", id='train-message-2'), style={'padding-right': '30px'}), 
                                    html.Td(html.H4(id='train-message'), style={'color': '#FF7F50'})
                                ]))]
                            , className= 'text-left')
                        ])
                    ]),
                    html.Div(style={'height': '25px'}),
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4('Related Links', className='text-center'),
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(html.Img(src="https://raw.githubusercontent.com/DericMei/First-Trial/f1741ff78a0b46f4241309472484b5327d42dfcb/DALL·E%202023-12-27%2014.59.15%20-%20Design%20a%20clear%20and%20striking%20logo%20featuring%20the%20initials%20'EM'%20for%20'Eric%20Mei%20Data%20Science'.%20Emphasize%20legibility%20and%20distinction%2C%20with%20the%20letters%20'E'%20a.png", 
                                                width='200', height='180'),
                                width=4, className='text-center'),
                                dbc.Col(
                                    dbc.Row(
                                    html.Table(
                                        [html.Tbody(html.Tr([
                                            html.Td(html.Strong(html.H4(html.A('LinkedIn', href='https://www.linkedin.com/in/zijiemei/'))), style={'color': '#FF7F50'}),
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.Strong(html.H4(html.A('GitHub', href='https://github.com/DericMei'))), style={'color': '#FF7F50'})
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.Strong(html.H4(html.A('Portfolio', href='https://zijiemei.com/'))), style={'color': '#FF7F50'})
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.Strong(html.H4(html.A('Medium', href='https://medium.com/@zijiemei'))), style={'color': '#FF7F50'})
                                        ]))] +
                                        [html.Tbody(html.Tr([
                                            html.Td(html.Strong(html.H4(html.A('Project Link', href='https://zijiemei.com/2024/01/28/bodyweight-dash-application/'))), style={'color': '#FF7F50'})
                                        ]))] 
                                    , className='text-center')
                                    ,)
                                , className= 'text-center', width=4),
                                dbc.Col(html.Img(src="https://raw.githubusercontent.com/DericMei/First-Trial/main/eric.png", 
                                                width='200', height='180'),
                                width=4, className='text-center'),
                            ])
                        ])
                    ]),
                ], width=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4('Body Weight Seasonality Trends by Prophet Model', className='text-center'),
                            dcc.Dropdown(
                                id = 'dropdown-ml',
                                options=[
                                    {'label': 'Weekly Seasonality', 'value': 'weekly'},
                                    {'label': 'Yearly Seasonality', 'value': 'yearly'},
                                    {'label': 'Overall Trend', 'value': 'all'},
                                ],
                                value='weekly' #default value
                            ),
                        ]),
                        dbc.CardBody(
                            dcc.Graph(id='ml', style={'padding': '0px', 'margin': '0px'})
                        )
                    ]),
                ], width=7)
            ]),
        ], style={'display': 'block'})
    ]),

], fluid=True)


###################
#### Callbacks ####
###################
# Callback to handle collapse
@app.callback(
    Output('collapse-part', "is_open"),
    [Input('collapse-button', "n_clicks")],
    [State('collapse-part', "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Callbacks to handle Password input
@app.callback(
    Output('password-output', 'children'),
    Output('card-1', 'style'),
    Output('message_2_1', 'style'),
    Output('message_2_2', 'style'),
    Output('message_2_3', 'style'),
    Output('card-2', 'style'),
    Output('message_1', 'style'),
    Output('weight-input', 'style'),
    Output('record-button', 'style'),
    Output('weight-record-status', 'style'),
    Output('message_3','style'),
    Output('button-train','style'),
    Output('train-message-2','style'),
    Output('train-message','style'),
    Input('unlock-button', 'n_clicks'),
    State('password-input', 'value')
)
def unlock_features(n_clicks, password):
    correct_password = os.getenv('PASSWORD')
    if n_clicks is not None and password == correct_password:
        return '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    elif n_clicks is not None and password != correct_password:
        return '', {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block', 'color': '#FF7F50'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    else:
        return '', {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block', 'color': '#FF7F50'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
app.layout['card-1'].style = {'display': 'block'}
app.layout['message_2_1'].style = {'display': 'block'}
app.layout['message_2_2'].style = {'display': 'none'}
app.layout['message_2_3'].style = {'display': 'block'}
app.layout['card-2'].style = {'display': 'none'}
app.layout['message_1'].style = {'display': 'none'}
app.layout['weight-input'].style = {'display': 'none'}
app.layout['record-button'].style = {'display': 'none'}
app.layout['weight-record-status'].style = {'display': 'none'}
app.layout['message_3'].style = {'display': 'block', 'color': '#FF7F50'}
app.layout['button-train'].style = {'display': 'none'}
app.layout['train-message-2'].style = {'display': 'none'}
app.layout['train-message'].style = {'display': 'none'}

# Weight record call back 
@app.callback(
    Output('weight-record-status', 'children'),
    Input('record-button', 'n_clicks'),
    State('weight-input', 'value')
)
def record_weight(n_clicks, weight):
    if n_clicks is not None and weight is not None:
        # Logic to handle weight recording and database interaction
        try:
            # Connect to the database
            with engine.connect() as conn:
                # Check for the last recorded weight
                last_record_query = text("SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1")
                result = conn.execute(last_record_query)
                last_record = result.fetchone()
                last_date, last_weight = last_record
                days_missing = (datetime.now().date() - last_date).days
                today_date_str = datetime.now().date().strftime('%Y-%m-%d')

                check_query = text("SELECT weight FROM weight_records WHERE date = :today_date")
                result = conn.execute(check_query, {'today_date': today_date_str})
                today_record = result.fetchone()

                if days_missing > 1:
                    for n in range(1, days_missing):
                        single_date_str = (last_date + timedelta(days=n)).strftime('%Y-%m-%d')
                        average_weight = (last_weight + weight) / 2
                        avg_instert_query = text("INSERT INTO weight_records (date, weight) VALUES (:date, :weight)")
                        conn.execute(avg_instert_query, {'date': single_date_str, 'weight': average_weight})
                        conn.commit()
                    update_query_1 = text("INSERT INTO weight_records (date, weight) Values (:date, :weight)")
                    conn.execute(update_query_1, {'date':today_date_str, 'weight':weight})
                    conn.commit()
                    return f'Weight recorded for {today_date_str}: {weight} kg! All Missing Weight Updated!'

                elif days_missing == 1:
                    update_query_2 = text("INSERT INTO weight_records (date, weight) Values (:date, :weight)")
                    conn.execute(update_query_2, {'date':today_date_str, 'weight':weight})
                    conn.commit()
                    return f'Weight recorded for {today_date_str}: {weight} kg!'
                
                elif today_record:
                    update_query_3 = text("UPDATE weight_records SET weight = :weight WHERE date = :today_date")
                    conn.execute(update_query_3, {'today_date': today_date_str, 'weight': weight})
                    conn.commit()
                    return f'Weight updated for {today_date_str}: {weight} kg!'

        except Exception as e:
            return f'An error occurred: {e}'
    else:
        return 'Record Weight Please! My Lord!'

# Callbacks to displays the last recorded weight, date, and the number of days recorded
@app.callback(
    [Output('last-recorded-date', 'children'),
     Output('last-recorded-weight', 'children')],
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def update_last_recorded_data(n):
    # Connect to the database and retrieve the last record
    with engine.connect() as conn:
        last_record_query = "SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1"
        last_record = pd.read_sql(last_record_query, conn)

    # Check if there is any record found
    if not last_record.empty:
        last_date = last_record.iloc[0]['date']
        last_weight = last_record.iloc[0]['weight']
        last_weight = format(last_weight, '.1f')
        return f'{last_date}', f'{last_weight} kg'
    else:
        return 'No records found', ''

# Callback to display the number of days recorded
@app.callback(
    Output('days-recording', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def update_days_recording(n):
    # Connect to the database and retrieve the last record date
    with engine.connect() as conn:
        last_record_query = "SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1"
        last_record = pd.read_sql(last_record_query, conn)
        total_records_query = "SELECT COUNT(*) FROM weight_records"
        total_records = pd.read_sql(total_records_query, conn)
        total_days = total_records.iloc[0,0]

    last_date = last_record['date'].iloc[0]
    days_missing = (datetime.now().date() - last_date).days

    if days_missing-1 <1:
        return f'{total_days} Days!!'
    else:
        return f'{days_missing-1} Days!!'
    
# Callback to supplement text for number of days recorded
@app.callback(
    Output('days-recording-text', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def update_days_recording_text(n):
    # Connect to the database and retrieve the last record date
    with engine.connect() as conn:
        last_record_query = "SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1"
        last_record = pd.read_sql(last_record_query, conn)

    last_date = last_record['date'].iloc[0]
    days_missing = (datetime.now().date() - last_date).days

    if days_missing-1 <1:
        return f'Bodyweight Tracked:'
    else:
        return f'Did not Track for:'
    
# Callback to display today's date
@app.callback(
    Output('week-status', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_week_info(n):
    today = datetime.now().date()
    return today

# Callback to display current weekly average weight
@app.callback(
    Output('current-week', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_current_avg(n):
    with engine.connect() as conn:
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week_str = start_of_week.strftime('%Y-%m-%d')
        query = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_week_str}'
                ORDER BY date
                """
        df = pd.read_sql(query, conn)
    avg_weights = df['weight'].mean()
    avg_weights = format(avg_weights, '.2f')
    return f'{avg_weights} kg'

# Callback to display last weekly average weight
@app.callback(
    Output('past-week', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_last_avg(n):
    today = datetime.today()
    start_of_last_week = today - timedelta(days=today.weekday() + 7)
    end_of_last_week = start_of_last_week + timedelta(days=6)
    start_of_last_week_str = start_of_last_week.strftime('%Y-%m-%d')
    end_of_last_week_str = end_of_last_week.strftime('%Y-%m-%d')
    with engine.connect() as conn:
        query = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_last_week_str}' AND date <= '{end_of_last_week_str}'
                ORDER BY date
                """
        df = pd.read_sql(query, conn)
    avg_weights = df['weight'].mean()
    avg_weights = format(avg_weights, '.2f')
    return f'{avg_weights} kg'

# Callback to display Difference between last week and this week
@app.callback(
    Output('weight-diff', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_weight_diff(n):
    with engine.connect() as conn:
        # Get current week df
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week_str = start_of_week.strftime('%Y-%m-%d')
        query_current = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_week_str}'
                ORDER BY date
                """
        df_current = pd.read_sql(query_current, conn)
        # Get last week df
        today = datetime.today()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        start_of_last_week_str = start_of_last_week.strftime('%Y-%m-%d')
        end_of_last_week_str = end_of_last_week.strftime('%Y-%m-%d')
        query_last = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_last_week_str}' AND date <= '{end_of_last_week_str}'
                ORDER BY date
                """
        df_last = pd.read_sql(query_last, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    weight_diff_display = format(abs(weight_diff), '.2f')
    return f'{weight_diff_display} kg'
    
# Callback to Display Calorie number
@app.callback(
    Output('calorie-amount', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_caloric_amount(n):
    with engine.connect() as conn:
        # Get current week df
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week_str = start_of_week.strftime('%Y-%m-%d')
        query_current = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_week_str}'
                ORDER BY date
                """
        df_current = pd.read_sql(query_current, conn)
        # Get last week df
        today = datetime.today()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        start_of_last_week_str = start_of_last_week.strftime('%Y-%m-%d')
        end_of_last_week_str = end_of_last_week.strftime('%Y-%m-%d')
        query_last = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_last_week_str}' AND date <= '{end_of_last_week_str}'
                ORDER BY date
                """
        df_last = pd.read_sql(query_last, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    daily_calorie = weight_diff*7700/7
    daily_calorie_display = format(daily_calorie, '.1f')
    return f'{daily_calorie_display} Kcal'
    
# Callback to Display Caloric Status
@app.callback(
    [Output('caloric-status', 'children'),
     Output('caloric-status', 'style')],
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_caloric_status(n):
    with engine.connect() as conn:
        # Get current week df
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week_str = start_of_week.strftime('%Y-%m-%d')
        query_current = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_week_str}'
                ORDER BY date
                """
        df_current = pd.read_sql(query_current, conn)
        # Get last week df
        today = datetime.today()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        start_of_last_week_str = start_of_last_week.strftime('%Y-%m-%d')
        end_of_last_week_str = end_of_last_week.strftime('%Y-%m-%d')
        query_last = f"""
                SELECT date, weight 
                FROM weight_records 
                WHERE date >= '{start_of_last_week_str}' AND date <= '{end_of_last_week_str}'
                ORDER BY date
                """
        df_last = pd.read_sql(query_last, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    weight_diff_abs = abs(weight_diff)
    if weight_diff_abs<0.1 and weight_diff <0:
        status_text = 'Small Deficit'
        status_style = {'color': '#28a745'}
    elif weight_diff_abs<0.1 and weight_diff >=0:
        status_text = 'Small Surplus'
        status_style = {'color': '#dc3545'} 
    elif weight_diff <0 and weight_diff_abs>=0.1:
        status_text = 'Deficit'
        status_style = {'color': '#28a745'}
    elif weight_diff >=0 and weight_diff_abs>=0.1:
        status_text = 'Surplus'
        status_style = {'color': '#dc3545'}
    return status_text, status_style

#### Plots Callbacks ####
# Callback to switch between day vs week and all data
@app.callback(
    [Output('day-vs-week-body', 'style'),
     Output('all-weight-data-body', 'style')],
    [Input('weight-data-dropdown', 'value')]
)
def toggle_card_bodies(selected_value):
    if selected_value == 'day_vs_week':
        return {'display': 'block'}, {'display': 'none'}
    elif selected_value == 'all_available':
        return {'display': 'none'}, {'display': 'block'}
    
# Callback: All data
@app.callback(
    Output('all', 'figure'),
    [Input('dropdown-all', 'value')]
)
def update_graph_all(selected_value):
    # Reading Data from database and create dataframes to plot
    with engine.connect() as conn:
        # Queries
        query_all = "SELECT date, weight FROM weight_records ORDER BY date"

        # Reading Queries for dataframes
        df = pd.read_sql(query_all, conn)

        # Making sure date is converted
        df['date'] = pd.to_datetime(df['date'])
    
    # DataFrames for plotting
    # 1. Daily
    df_daily = df.copy()

    # 2. Weekly
    df_weekly = df.copy()
    df_weekly.set_index('date', inplace=True)
    df_weekly = df_weekly['weight'].resample('W').mean().reset_index()
    df_weekly['week'] = df_weekly['date']

    # 3. Monthly
    df_monthly = df.copy()
    df_monthly.set_index('date', inplace=True)
    df_monthly = df_monthly['weight'].resample('ME').mean().reset_index()
    df_monthly['month'] = df_monthly['date']

    # 4. Seasonally
    df_seasonally = df.copy()
    df_seasonally.set_index('date', inplace=True)
    df_seasonally = df_seasonally['weight'].resample('QE').mean().reset_index()
    df_seasonally['season'] = df_seasonally['date']

    # plot 1 (Daily)
    if selected_value == 'daily':
        fig = px.line(df_daily, x='date', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Daily Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 2 (Weekly Avg)
    elif selected_value == 'weekly':
        fig = px.line(df_weekly, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 3 (Monthly Avg)
    elif selected_value == 'montly':
        fig = px.line(df_monthly, x='month', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Monthly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 4 (Seasonaly Avg)
    elif selected_value == 'seasonaly':
        fig = px.line(df_seasonally, x='season', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Seaonal Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig

# Callback: Daily
@app.callback(
    Output('daily', 'figure'),
    [Input('dropdown-daily', 'value')]
)
def update_graph_daily(selected_value):
    # Reading Data from database and create dataframes to plot
    with engine.connect() as conn:
        # Date time Calculations
        # 1. Start of the current week (Monday)
        start_of_week = datetime.now().date() - timedelta(days=datetime.now().weekday())
        # 2. Last week
        today = datetime.now().date()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)  # Subtract current weekday + 7 days
        end_of_last_week = start_of_last_week + timedelta(days=6)  # Last day of the last week (Sunday)
        # 3. Start of the current month
        start_of_month = datetime.now().date().replace(day=1)
        end_of_month = start_of_month + MonthEnd(0)
        # 4. Start of the first day of 5 weeks ago
        start_of_five_weeks_ago = today - timedelta(days=today.weekday() + 35)
        # 5. Start of the first day of 3 months ago
        start_of_three_month_ago = start_of_month - relativedelta(months=2)
        # 6. Start of the first day of 6 months ago
        start_of_six_month_ago = start_of_month - relativedelta(months=5)
        # 7. Start of the first day of 12 months ago
        start_of_one_year_ago = start_of_month - relativedelta(months=11)

        # Queries
        query_current_week = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_week}' ORDER BY date"
        query_last_week = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_last_week}' AND date <= '{end_of_last_week}' ORDER BY date"
        query_current_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_month}' ORDER BY date"
        query_past_six_weeks = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_five_weeks_ago}' ORDER BY date"
        query_past_three_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_three_month_ago}' ORDER BY date"
        query_past_six_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_six_month_ago}' ORDER BY date"
        query_past_one_year = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_one_year_ago}' ORDER BY date"

        # Reading Queries for dataframes
        df_1 = pd.read_sql(query_current_week, conn)
        df_2 = pd.read_sql(query_last_week, conn)
        df_3 = pd.read_sql(query_current_month, conn)
        df_4 = pd.read_sql(query_past_six_weeks, conn)
        df_5 = pd.read_sql(query_past_three_month, conn)
        df_6 = pd.read_sql(query_past_six_month, conn)
        df_7 = pd.read_sql(query_past_one_year, conn)

        # Making sure date is converted
        df_1['date'] = pd.to_datetime(df_1['date'])
        df_2['date'] = pd.to_datetime(df_2['date'])
        df_3['date'] = pd.to_datetime(df_3['date'])
        df_4['date'] = pd.to_datetime(df_4['date'])
        df_5['date'] = pd.to_datetime(df_5['date'])
        df_6['date'] = pd.to_datetime(df_6['date'])
        df_7['date'] = pd.to_datetime(df_7['date'])
    
    # DataFrames for plotting
    # 1
    week_dates = pd.date_range(start=start_of_week, periods=7, freq='D')
    df_current_week = pd.DataFrame(week_dates, columns=['date']).merge(df_1, on='date', how='left')
    # 2
    last_week_dates = pd.date_range(start=start_of_last_week, periods=7, freq='D')
    df_last_week = pd.DataFrame(last_week_dates, columns=['date']).merge(df_2, on='date', how='left')
    # 3
    month_dates = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
    df_current_month = pd.DataFrame(month_dates, columns=['date']).merge(df_3, on='date', how='left')
    # 4
    six_weeks_dates = pd.date_range(start=start_of_five_weeks_ago, periods=42, freq='D')
    df_six_weeks = pd.DataFrame(six_weeks_dates, columns=['date']).merge(df_4, on='date', how='left')
    # 5
    three_month_dates = pd.date_range(start=start_of_three_month_ago, end=end_of_month, freq='D')
    df_three_month = pd.DataFrame(three_month_dates, columns=['date']).merge(df_5, on='date', how='left')
    # 6
    six_month_dates = pd.date_range(start=start_of_six_month_ago, end=end_of_month, freq='D')
    df_six_month = pd.DataFrame(six_month_dates, columns=['date']).merge(df_6, on='date', how='left')
    # 7
    one_year_dates = pd.date_range(start=start_of_one_year_ago, end=end_of_month, freq='D')
    df_12_month = pd.DataFrame(one_year_dates, columns=['date']).merge(df_7, on='date', how='left')

    # plot 1 (current week)
    if selected_value == 'current_week':
        fig = px.line(df_current_week, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%a'),
            yaxis=dict(range=[84,87])
            )
        return fig
    
    # plot 2 (last week)
    elif selected_value == 'last_week':
        fig = px.line(df_last_week, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%a'),
            yaxis=dict(range=[84,87])
        )
        return fig
    
    # plot 3 (current month)
    elif selected_value =='current_month':
        fig = px.line(df_current_month, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%d')
        )
        return fig
    
    # plot 4 (6 weeks)
    elif selected_value == '6_weeks':
        fig = px.line(df_six_weeks, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        return fig
    
    # plot 5 (past 3 months)
    elif selected_value == 'three_month':
        fig = px.line(df_three_month, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%y-%m-%d')
        )
        return fig
    
    # plot 6 (Past 6 Months)
    elif selected_value == 'six_month':
        fig = px.line(df_six_month, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%y-%m-%d')
        )
        return fig
    
    # plot 7 (Past Year)
    elif selected_value == 'one_year':
        fig = px.line(df_12_month, x='date', y='weight')
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Weight (kg)',
            margin=dict(t=10, l=0, r=0, b=0),
            xaxis=dict(tickformat='%y-%m-%d')
        )
        return fig

# Callback: Weekly Avg
@app.callback(
    Output('weekly', 'figure'),
    [Input('dropdown-weekly', 'value')]
)
def update_graph_weekly_avg(selected_value):
    # Reading Data from database and create dataframes to plot
    with engine.connect() as conn:
        # Date time Calculations
        # 1. Start of the current month
        start_of_month = datetime.now().date().replace(day=1)
        end_of_month = start_of_month + MonthEnd(0)
        # 2. Start of the first day of 5 weeks ago 
        today = datetime.now().date()
        start_of_five_weeks_ago = today - timedelta(days=today.weekday() + 35)
        # 3. Start of the first day of 3 months ago
        start_of_three_month_ago = start_of_month - relativedelta(months=2)
        # 4. Start of the first day of 6 months ago
        start_of_six_month_ago = start_of_month - relativedelta(months=5)
        # 5. Start of the first day of 12 months ago
        start_of_one_year_ago = start_of_month - relativedelta(months=11)
        
        # Queries
        query_current_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_month}' ORDER BY date"
        query_past_six_weeks = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_five_weeks_ago}' ORDER BY date"
        query_past_three_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_three_month_ago}' ORDER BY date"
        query_past_six_month = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_six_month_ago}' ORDER BY date"
        query_past_one_year = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_one_year_ago}' ORDER BY date"

        # Reading Queries for dataframes
        df_1 = pd.read_sql(query_current_month, conn)
        df_2 = pd.read_sql(query_past_six_weeks, conn)
        df_3 = pd.read_sql(query_past_three_month, conn)
        df_4 = pd.read_sql(query_past_six_month, conn)
        df_5 = pd.read_sql(query_past_one_year, conn)

        # Making sure date is converted
        df_1['date'] = pd.to_datetime(df_1['date'])
        df_2['date'] = pd.to_datetime(df_2['date'])
        df_3['date'] = pd.to_datetime(df_3['date'])
        df_4['date'] = pd.to_datetime(df_4['date'])
        df_5['date'] = pd.to_datetime(df_5['date'])

    # DataFrames for plotting
    # 1
    month_dates = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
    df_current_month = pd.DataFrame(month_dates, columns=['date']).merge(df_1, on='date', how='left')
    df_current_month.set_index('date', inplace=True)
    df_current_month = df_current_month['weight'].resample('W').mean().reset_index()
    df_current_month['week'] = df_current_month['date'].dt.strftime('Week %U')
    
    # 2
    six_weeks_dates = pd.date_range(start=start_of_five_weeks_ago, periods=42, freq='D')
    df_six_weeks = pd.DataFrame(six_weeks_dates, columns=['date']).merge(df_2, on='date', how='left')
    df_six_weeks.set_index('date', inplace=True)
    df_six_weeks = df_six_weeks['weight'].resample('W').mean().reset_index()
    df_six_weeks['week'] = df_six_weeks['date']
    # 3
    three_month_dates = pd.date_range(start=start_of_three_month_ago, end=end_of_month, freq='D')
    df_three_month = pd.DataFrame(three_month_dates, columns=['date']).merge(df_3, on='date', how='left')
    df_three_month.set_index('date', inplace=True)
    df_three_month = df_three_month['weight'].resample('W').mean().reset_index()
    df_three_month['week'] = df_three_month['date']
    # 4
    six_month_dates = pd.date_range(start=start_of_six_month_ago, end=end_of_month, freq='D')
    df_six_month = pd.DataFrame(six_month_dates, columns=['date']).merge(df_4, on='date', how='left')
    df_six_month.set_index('date', inplace=True)
    df_six_month = df_six_month['weight'].resample('W').mean().reset_index()
    df_six_month['week'] = df_six_month['date']
    # 5
    one_year_dates = pd.date_range(start=start_of_one_year_ago, end=end_of_month, freq='D')
    df_12_month = pd.DataFrame(one_year_dates, columns=['date']).merge(df_5, on='date', how='left')
    df_12_month.set_index('date', inplace=True)
    df_12_month = df_12_month['weight'].resample('W').mean().reset_index()
    df_12_month['week'] = df_12_month['date']

    # plot 1 (current month)
    if selected_value == 'current_month':
        fig = px.line(df_current_month, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        )
        return fig
    
    # plot 2 (Past 6 Weeks)
    elif selected_value == '6_weeks':
        fig = px.line(df_six_weeks, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 3 (Past 3 Months)
    elif selected_value == 'three_month':
        fig = px.line(df_three_month, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 4 (Past 6 Months)
    elif selected_value == 'six_month':
        fig = px.line(df_six_month, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig
    
    # plot 5 (Past Year)
    elif selected_value == 'one_year':
        fig = px.line(df_12_month, x='week', y='weight')
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
        )
        return fig

# Callbacks for Machine Learning Part
'''
# Callback to switch between 2 models
@app.callback(
    [Output('prophet-body', 'style'),
     Output('lstm-body', 'style')],
    [Input('machine-learning-dropdown', 'value')]
)
def toggle_card_bodies(selected_value):
    if selected_value == 'prophet':
        return {'display': 'block'}, {'display': 'none'}
    elif selected_value == 'lstm':
        return {'display': 'none'}, {'display': 'block'}
'''

# Callback for Machine Learning trend plots
@app.callback(
    Output('ml', 'figure'),
    [Input('dropdown-ml', 'value')]
)
def update_graph(selected_trend):
    # Initial setup
    # load the model using function defined
    model = load_model()
    # Define future dataframe for prediction
    future = model.make_future_dataframe(periods=7)
    # Generate forecast dataframe by predicting using the model
    forecast = model.predict(future)

    if selected_trend == 'weekly':
        # prepare data
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        forecast['day_of_week'] = forecast['ds'].dt.day_name()
        weekly_means = forecast.groupby('day_of_week')['weekly'].mean().reindex(ordered_days).reset_index()
        # plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_means['day_of_week'], y=weekly_means['weekly'], mode='lines'))
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Unit (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
        )
        return fig
    elif selected_trend == 'yearly':
        # prepare data
        yearly = forecast[['ds', 'yearly']].dropna()
        yearly['ds'] = pd.to_datetime(yearly['ds'])
        yearly['day_of_year'] = forecast['ds'].dt.dayofyear
        yearly = yearly.groupby('day_of_year')['yearly'].mean().reset_index()

        month_ticks = list(range(1, 366, 30))  # Start of each month
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['day_of_year'], y=yearly['yearly'], mode='lines'))
        fig.update_traces(connectgaps=True)
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Unit (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(
        title='Month of Year',
        tickmode='array',
        tickvals=month_ticks,
        ticktext=month_labels
        ),
        )
        return fig
    elif selected_trend == 'all':
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines'))
        fig.update_traces(connectgaps=True)
        fig.update_layout(
        xaxis_title='',
        yaxis_title='Unit (kg)',
        margin=dict(t=10, l=0, r=0, b=0))
        return fig

# Callback to display day of week
@app.callback(
    Output('day-of-week', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_day_of_week(n):
    current_date = datetime.now()
    day_of_week = current_date.strftime('%A')
    return day_of_week

# Callback to display forecasted today's bodyweight
@app.callback(
    Output('today-forecast', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_today_forecast(n):
    model = load_model()
    today = pd.DataFrame({'ds': [datetime.now().date()]})
    forecast = model.predict(today)
    today_forecast_value = forecast['yhat'].iloc[0]
    return f'{today_forecast_value:.2f} kg'

# Callback to display recorded bodyweight of today
@app.callback(
    Output('weight-today', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_today_record(n):
    today = datetime.now().date()
    with engine.connect() as conn:
        today_record_query = f"SELECT date, weight FROM weight_records WHERE date = '{today}'"
        df_today_record = pd.read_sql(today_record_query, conn)
    if not df_today_record.empty:
        last_weight = df_today_record.iloc[0]['weight']
        last_weight = format(last_weight, '.1f')
        return f"{last_weight} kg"
    else:
        return "You Forgot to Record!"
    
# Callback to display model training status
@app.callback(
    Output('model-train-time', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_model_not_train_time(n):
    today = datetime.now().date()
    with engine.connect() as conn:
        last_record = text("SELECT training_date FROM models ORDER BY training_date DESC LIMIT 1")
        result = conn.execute(last_record)
        record = result.fetchone()
        training_date = record[0]
        days_since_trained = (today - training_date).days
    if days_since_trained == 0:
        return 'Just Trained Today!'
    else:
        return f'{days_since_trained} Days'

# Callback to train the model with press of a button
@app.callback(
    Output('train-message', 'children'),
    [Input('button-train', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_message(n_clicks, n):

    if n_clicks is None:
        return 'Takes about 2 seconds~'
    
    else:
        # Get all data from database
        with engine.connect() as conn:
            query_all = "SELECT date, weight FROM weight_records ORDER BY date"
            df = pd.read_sql(query_all, conn)

        # Prepare data to train
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df_prophet = df.reset_index().rename(columns={'date': 'ds', 'weight': 'y'})

        # Train the model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        model.fit(df_prophet)

        # Prepare model to load into database
        training_date = datetime.now().date()
        serialized_model = pickle.dumps(model)

        # Insert newly trained model into database
        with engine.connect() as conn:
            insert_query = text("""
                INSERT INTO models (model_data, training_date)
                VALUES (:model_data, :training_date)
            """)
            conn.execute(insert_query, {
                'model_data': serialized_model,
                'training_date': training_date
            })
            conn.commit()

        return "Training Complete!"

'''
# LSTM part
# Callback to generate prediction table
@app.callback(
    Output('lstm-table', 'data'),
    Output('lstm-table', 'columns'),
    [Input('interval-component', 'n_intervals')]
)
def load_data(n):
    # Get all data from database
    with engine.connect() as conn:
        query_all = "SELECT date, weight FROM weight_records ORDER BY date"
        df = pd.read_sql(query_all, conn)
        df['date'] = pd.to_datetime(df['date'])

    # Generate predictions using the LSTM model
    df_predictions = predict_future_weights_with_dates(df, model_lstm, scaler, future_days=7)

    # Format date and weight columns
    df_predictions['date'] = df_predictions['date'].dt.strftime('%Y-%m-%d')
    df_predictions['predicted_weight'] = df_predictions['predicted_weight'].apply(lambda x: f'{x:.2f}')

    # Prepare the data for the DataTable
    data = df_predictions.to_dict('records')
    columns = [{"name": "Date", "id": "date"}, {"name": "Weight (kg)", "id": "predicted_weight"}]
    return data, columns

# Callback to generate prediction plot from the lstm model
@app.callback(
    Output('lstm-plot', 'figure'),
    [Input('interval-component', 'n_intervals')] 
)
def update_graph(n_intervals):
    with engine.connect() as conn:
        query_all = "SELECT date, weight FROM weight_records ORDER BY date"
        df = pd.read_sql(query_all, conn)
        df['date'] = pd.to_datetime(df['date'])

    # Predict weights for each date in the DataFrame
    df_predictions = predict_all_weights(df, model_lstm, scaler)

    # Create the Plotly figure and add the actual and predicted weight traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_predictions['date'], 
        y=df_predictions['actual_weight'], 
        mode='lines', 
        name='Actual Weight', 
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_predictions['date'], 
        y=df_predictions['predicted_weight'], 
        mode='lines', 
        name='Predicted Weight', 
        line=dict(color='red')
    ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Weight (kg)',
        hovermode='x unified'
    )
    return fig
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)