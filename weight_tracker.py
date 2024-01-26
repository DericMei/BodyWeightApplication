# Import necessary libraries
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import psycopg2
import os
from datetime import datetime
from datetime import timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import dash_bootstrap_components as dbc

# This is to load local environmental variables for testing the application
# load_dotenv()
# database_url = os.getenv('DATABASE_URL')

# for deployment
database_url = os.environ.get('DATABASE_URL')
# Check if the URL starts with "postgres://" this is a weird error with postgresql on heroku
if database_url.startswith('postgres://'):
    # Replace it with "postgresql://"
    database_url = 'postgresql://' + database_url[len('postgres://'):]

# Connect to database
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

    # Spacer
    html.Div(style={'height': '10px'}),

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
                                    html.H5(id='week-status', className='text-left'),
                                    html.H5(id='past-week', className='text-left'),
                                    html.H5(id='current-week', className='text-left')
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5(id='weight-diff', className='text-left'),
                                    html.H5(id='calorie-status', className='text-left'),
                                    html.H5(id='advice', className='text-left')
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
                        width=3, className='text-center'),
                        dbc.Col(
                            html.Div(id='weight-record-status'),
                        width=5, className='text-left')
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
                dbc.CardHeader(html.H4("Daily Weight Last Week"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='daily-weight-last-week', style={'padding': '0px', 'margin': '0px'})
                ],style={'padding': '0px'})
            ]), 
        width=4),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Daily Weight Current Week"), className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='daily-weight-current-week', style={'padding': '0px', 'margin': '0px'})
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
    avg_weights = format(avg_weights, '.2f')
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
    avg_weights = format(avg_weights, '.2f')
    return f'Last Week Average: {avg_weights}kg'

# Callback to display Difference between last week and this week
@app.callback(
    Output('weight-diff', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_weight_diff(n):
    with engine.connect() as conn:
        query_last = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) - interval '1 week' AND date < date_trunc('week', current_date) ORDER BY date"
        df_last = pd.read_sql(query_last, conn)
        query_current = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) ORDER BY date"
        df_current = pd.read_sql(query_current, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    weight_diff_display = format(abs(weight_diff), '.2f')
    if weight_diff <0:
        return f'You Have Lost: {weight_diff_display}kg Comparing to Last Week! Nice!'
    elif weight_diff >0:
        return f'You Have Gained: {weight_diff_display}kg Comparing to Last Week!'
    else:
        return f'No Difference Comparing to Last Week! Maintaining~'
    
# Callback to Display Caloric Status
@app.callback(
    Output('calorie-status', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def get_caloric_status(n):
    with engine.connect() as conn:
        query_last = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) - interval '1 week' AND date < date_trunc('week', current_date) ORDER BY date"
        df_last = pd.read_sql(query_last, conn)
        query_current = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) ORDER BY date"
        df_current = pd.read_sql(query_current, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    daily_calorie = abs(weight_diff)*7700/7
    daily_calorie_display = format(abs(daily_calorie), '.0f')
    if weight_diff <0:
        return f'Estimated Daily Caloric Deficit: {daily_calorie_display}Kcal'
    elif weight_diff >0:
        return f'Estimated Daily Caloric Surplus: {daily_calorie_display}Kcal'
    else:
        return f'No Difference Comparing to Last Week! Maintaining~'
    
# Callback to Display Advice
@app.callback(
    Output('advice', 'children'),
    [Input('interval-component', 'n_intervals')]  # This input is triggered when the page loads
)
def advice(n):
    with engine.connect() as conn:
        query_last = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) - interval '1 week' AND date < date_trunc('week', current_date) ORDER BY date"
        df_last = pd.read_sql(query_last, conn)
        query_current = "SELECT date, weight FROM weight_records WHERE date>=date_trunc('week', current_date) ORDER BY date"
        df_current = pd.read_sql(query_current, conn)
    avg_weights_last = df_last['weight'].mean()
    avg_weights_current = df_current['weight'].mean()
    weight_diff = avg_weights_current-avg_weights_last
    weight_diff_abs = abs(weight_diff)
    if weight_diff <0 and weight_diff_abs<0.5:
        return f'Advice: Losing Weight, but not Enough, COME ON Eric!' 
    elif weight_diff <0 and weight_diff_abs>=0.5:
        return f'Advice: Good Job My Lord! You Are The God!'
    elif weight_diff >=0:
        return f'Advice: What Are You Doing?? Stop Eating Crap!!!'

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
    fig.update_layout(yaxis=dict(range=[81,87]))
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
    fig.update_layout(yaxis=dict(range=[81,87]))
    return fig

# Callback to display daily weight for Current Week
@app.callback(
    Output('daily-weight-current-week', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        # Calculate the start of the current week (Monday)
        start_of_week = datetime.now().date() - timedelta(days=datetime.now().weekday())
        query = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_week}' ORDER BY date"
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])
    # Create a DataFrame for the entire week
    week_dates = pd.date_range(start=start_of_week, periods=7, freq='D')
    df_week = pd.DataFrame(week_dates, columns=['date']).merge(df, on='date', how='left')
    # Plotting
    fig = px.line(df_week, x='date', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%a')
    )
    fig.update_layout(yaxis=dict(range=[84,86.5]))
    return fig

# Callback to display daily weight for Last Week
@app.callback(
    Output('daily-weight-last-week', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        # Calculate the start of the last week (previous Monday)
        today = datetime.now().date()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)  # Subtract current weekday + 7 days
        end_of_last_week = start_of_last_week + timedelta(days=6)  # Last day of the last week (Sunday)

        # SQL query to get weights from the last week
        query = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_last_week}' AND date <= '{end_of_last_week}' ORDER BY date"
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])

    # Create a DataFrame for the entire last week
    last_week_dates = pd.date_range(start=start_of_last_week, periods=7, freq='D')
    df_last_week = pd.DataFrame(last_week_dates, columns=['date']).merge(df, on='date', how='left')

    # Plotting
    fig = px.line(df_last_week, x='date', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickformat='%a')  # Format x-axis labels to show day of the week
    )
    fig.update_layout(yaxis=dict(range=[84,86.5]))
    return fig

# Callback to display weekly average for past 5 weeks including current week
@app.callback(
    Output('weekly-avg-6weeks', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    with engine.connect() as conn:
        # Calculate the start of the week five weeks ago
        today = datetime.now().date()
        start_of_five_weeks_ago = today - timedelta(days=today.weekday() + 35)

        # SQL query to get weights from the last five weeks
        query = f"SELECT date, weight FROM weight_records WHERE date >= '{start_of_five_weeks_ago}' ORDER BY date"
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])

    # Determine the current week number
    current_week_number = (today - start_of_five_weeks_ago).days // 7 + 1

    # Calculate the relative week number
    df['relative_week'] = ((df['date'] - pd.Timestamp(start_of_five_weeks_ago)).dt.days // 7) + 1

    # Map the relative week number to custom labels
    week_labels = {i: f"Week {i}" for i in range(1, current_week_number)}
    week_labels[current_week_number] = "Current"
    df['week_label'] = df['relative_week'].map(week_labels)

    # Group by relative week and calculate weekly average
    df_weekly_avg = df.groupby('week_label')['weight'].mean().reset_index()

    # Ensure the order is Week 1 to Week 5, then Current Week
    ordered_labels = [f"Week {i}" for i in range(1, current_week_number)] + ["Current"]
    df_weekly_avg['week_label'] = pd.Categorical(df_weekly_avg['week_label'], categories=ordered_labels, ordered=True)
    df_weekly_avg.sort_values('week_label', inplace=True)

    # Plotting
    fig = px.line(df_weekly_avg, x='week_label', y='weight')
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Average Weight (kg)',
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(tickangle=45)
    )
    fig.update_layout(yaxis=dict(range=[84,86.5]))
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)