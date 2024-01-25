# Import necessary libraries
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import psycopg2
import os
from datetime import datetime
from datetime import timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import dash_bootstrap_components as dbc

# This is to load local environmental variables for testing the application
load_dotenv()

# Connect to database
database_url = os.getenv('DATABASE_URL')
engine = create_engine(database_url)

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

    # Title of the application
    dbc.Card(dbc.CardHeader(html.H1(html.Strong("Eric Mei BodyWeight Magic!"), className='text-center'))),

    # Spacer
    html.Div(style={'height': '10px'}),

    # Basic Info
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Weight Tracking Informations", className='text-center')),
                dbc.CardBody([
                    html.H4(id='days-recording', className='text-left'),
                    html.H5(id='last-recorded-date', className='text-left'),
                    html.H5(id='last-recorded-weight', className='text-left')
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
                                    html.H5(id='week-status', className='text-center'),
                                    html.H5(id='current-week', className='text-center'),
                                    html.H5(id='past-week', className='text-center')
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5('weight-diff-status', className='text-center'),
                                    html.H5('convert-calories', className='text-center'),
                                    html.H5('advice', className='text-center')
                                ])
                            ])
                        ], width=6),
                    ])
                ])
            ]), width=8)
    ]),

    # Spacer
    html.Div(style={'height': '10px'}),

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
                    html.Div(style={'height': '10px'}),
                    # output component to display messages about the status of password entering
                    dbc.Row(html.Div(id='password-output')),
                    #Display text hint for incorrect password
                    dbc.Row(html.H4("You! Shall! Not! Pass!", id='message_pass', className='text-center'))
                ]), 
            ]),
        width=6),

        # Weight Recording Parts
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Welcome Home! Lord Eric, Please Record Your Weight:", id='message_1', className='text-center')),
                dbc.CardBody([
                    dbc.Row(html.H5("Today's Date: " + datetime.now().strftime('%Y-%m-%d'), id='message_2', className='text-left')),
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
                        width=4, className='text-center'),
                        dbc.Col(
                            html.Div(id='weight-record-status'),
                        width=4, className='text-left')
                    ], className='align-items-center')
                ])
            ]),
        width=6, className='text-left')
    ]),

    # Spacer
    html.Div(style={'height': '10px'}),

    # Visualizations: Important Metrics for myself
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Daily Weight for Past Week"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='daily-weight-week', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'})
            ]), 
        width=4),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Daily Weight for Past 6 Weeks"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='daily-weight-6weeks', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=4),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Weekly Average for Past 6 Weeks"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='weekly-avg-6weeks', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=4)
    ]),

    # Spacer
    html.Div(style={'height': '10px'}),

    # Visualizations: For comparison
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Weekly Average Trend for Past 3 Months"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='week-3-month', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'})
            ]), 
        width=6),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Daily Weight for Past 3 Months"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='day-3-month', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=6),
    ]),

    # Spacer
    html.Div(style={'height': '10px'}),

    # Visualizations: Nice to know
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Weekly Average Trend for Past Year"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='week-year', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=6),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Weekly Average Trend for All Data"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='week-all', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=6),
    ]),

    # Spacer
    html.Div(style={'height': '10px'}),

    # Visualizations: Just for fun Overall Body weight trend
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("All Body Weight Records Data Plot"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='all-weight-plot', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'} )
            ]), width=12)
    ]),

], fluid=True)

# Callbacks to handle Password input
@app.callback(
    Output('password-output', 'children'),
    Output('message_pass', 'style'),
    Output('message_1', 'style'),
    Output('message_2', 'style'),
    Output('weight-input', 'style'),
    Output('record-button', 'style'),
    Output('weight-record-status', 'style'),
    Input('unlock-button', 'n_clicks'),
    State('password-input', 'value')
)
def unlock_features(n_clicks, password):
    correct_password = os.getenv('PASSWORD')
    if n_clicks is not None and password == correct_password:
        return '', {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    elif n_clicks is not None and password != correct_password:
        return '', {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    else:
        return '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
app.layout['message_1'].style = {'display': 'none'}
app.layout['message_2'].style = {'display': 'none'}
app.layout['weight-input'].style = {'display': 'none'}
app.layout['record-button'].style = {'display': 'none'}
app.layout['weight-record-status'].style = {'display': 'none'}

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
            conn = engine.connect()
            
            # Check for the last recorded weight
            last_record_query = "SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1"
            last_record = pd.read_sql(last_record_query, conn)
            
            # If there's no last record, we'll just insert today's weight
            if last_record.empty:
                conn.execute("INSERT INTO weights (date, weight) VALUES (%s, %s)", 
                             (datetime.now().date().strftime('%Y-%m-%d'), weight))
                conn.close()
                return f'Recorded new weight for today: {weight} kg'
            
            # Calculate the average value for missing days
            last_date = last_record['date'].iloc[0]
            days_missing = (datetime.now().date() - last_date).days
            if days_missing > 1:
                for single_date in (last_date + timedelta(days=n) for n in range(1, days_missing)):
                    average_weight = (last_record['weight'].iloc[0] + weight) / 2
                    conn.execute("INSERT INTO weights (date, weight) VALUES (%s, %s)", 
                                 (single_date, average_weight))
            
            # Finally, insert today's weight
            conn.execute("INSERT INTO weights (date, weight) VALUES (%s, %s)", 
                         (datetime.now().date(), weight))
            conn.close()
            return f'Weight recorded for {datetime.now().date()}: {weight} kg'
        
        except Exception as e:
            conn.close()
            return f'An error occurred: {e}'
        
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
        return f'Last Recorded Date: {last_date}', f'Last Recorded Weight: {last_weight} kg'
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
        return f'Lord Eric! You have been recording your Body Weight for {total_days} Days!'
    else:
        return f'What! You forgot to record for {days_missing-1} Days!! Come on!!'
    
# Callback to display the current day of the week
@app.callback(
    Output('week-status', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_week_info(n):
    today = datetime.now().date()
    week_day = today.isocalendar()[2]
    return f'You are at the {week_day} Day of this current week.'

# Callback to display current weekly average weight
@app.callback(
    Output('current-week', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_current_avg(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) ORDER BY date"
        df = pd.read_sql(query, conn)
    avg_weights = df['weight'].mean()
    avg_weights = format(avg_weights, '.1f')
    return f'Current Week Average: {avg_weights}kg'

# Callback to display last weekly average weight
@app.callback(
    Output('past-week', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_last_avg(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) - interval '1 week' AND date < date_trunc('week', current_date) ORDER BY date"
        df = pd.read_sql(query, conn)
    avg_weights = df['weight'].mean()
    avg_weights = format(avg_weights, '.1f')
    return f'Last Week Average: {avg_weights}kg'

# Callback to display all weight trend plot
@app.callback(
    Output('all-weight-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records ORDER BY date"
        df = pd.read_sql(query, conn)
    fig = px.line(df, x='date', y='weight')
    fig.update_layout(
        xaxis_title='', yaxis_title='Weight (kg)',
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
        )
    )
    fig.update_layout(title_text='', margin=dict(t=10, l=25, r=15, b=30))
    return fig

# Callback to display weekly average for past 3 months
@app.callback(
    Output('week-3-month', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '3 months' ORDER BY date"
        df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    weekly_averages = df['weight'].resample('W').mean().reset_index()
    weekly_averages['week'] = weekly_averages['date'].dt.strftime('Week %U')
    fig = px.line(weekly_averages, x='week', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display weekly average for past year
@app.callback(
    Output('week-year', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '1 year' ORDER BY date"
        df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    weekly_averages = df['weight'].resample('W').mean().reset_index()
    weekly_averages['week'] = weekly_averages['date'].dt.strftime('Week %U')
    fig = px.line(weekly_averages, x='week', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display weekly average for all data
@app.callback(
    Output('week-all', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records ORDER BY date"
        df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    weekly_averages = df['weight'].resample('W').mean().reset_index()
    weekly_averages['week'] = weekly_averages['date'].dt.strftime('Week %U, %Y')
    fig = px.line(weekly_averages, x='week', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display daily weight for past 3 months
@app.callback(
    Output('day-3-month', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '3 months' ORDER BY date"
        df = pd.read_sql(query, conn)
    fig = px.line(df, x='date', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display daily weight for past week
@app.callback(
    Output('daily-weight-week', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '1 week' ORDER BY date"
        df = pd.read_sql(query, conn)
    fig = px.line(df, x='date', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display daily weight for past 6 weeks
@app.callback(
    Output('daily-weight-6weeks', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '6 weeks' ORDER BY date"
        df = pd.read_sql(query, conn)
    fig = px.line(df, x='date', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Callback to display weekly average for past 6 weeks
@app.callback(
    Output('weekly-avg-6weeks', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        query = "SELECT date, weight FROM weight_records WHERE date >= current_date - interval '6 weeks' ORDER BY date"
        df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    weekly_averages = df['weight'].resample('W').mean().reset_index()
    weekly_averages['week'] = weekly_averages['date'].dt.strftime('Week %U')
    fig = px.line(weekly_averages, x='week', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weekly Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0)
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)