# Import necessary libraries
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import psycopg2
import os
import datetime

# Connect to database
def connect_to_db():
    database_url = os.environ['postgres://ldxyurgvquzdsi:5464073634d12d89c3d98d4d47f8e0de542d17867b3d2aa2157668a59d50e80c@ec2-52-54-200-216.compute-1.amazonaws.com:5432/d77ae5bt6775c2']
    conn = psycopg2.connect(database_url, sslmode='require')
    return conn

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for Heroku

# Define Dash app layout
app.layout = html.Div([
    html.H1("Eric Mei BodyWeight Magic Tool!"),

    dcc.DatePickerSingle(
        id='date-picker',
        date=datetime.now().date(),
        display_format='YYYY-MM-DD'
    ),
    
    dcc.Input(
        id='weight-input',
        type='number',
        min=70, max=95, step=0.1
    ),
    
    html.Button('Record Weight', id='record-button'),
    
    html.Div(id='output-container'),
    
    dcc.Graph(id='weight-graph')
])

# Callbacks to handle user interaction
@app.callback(
    Output('output-container', 'children'),
    Output('weight-graph', 'figure'),
    Input('record-button', 'n_clicks'),
    [Input('date-picker', 'date'), Input('weight-input', 'value')]
)
def record_weight(n_clicks, date, weight):
    if n_clicks is not None and date is not None and weight is not None:
        # Logic to handle weight recording and database interaction

        # Generate a plotly figure based on the updated data
        fig = px.line(...)  # Replace with your Plotly graph generation logic

        return f'Weight recorded for {date}: {weight} kg', fig
    
    return '', px.line()  # Empty plot as a placeholder

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)














####################


# Function to get the last recored weight
def get_last_recorded_weight(conn):
    query = "SELECT weight FROM weight_records ORDER BY date DESC LIMIT 1"
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()
    return result[0] if result else None

# Function to insert or update weight data
def fill_missing_dates(conn, new_date, new_weight):
    cur = conn.cursor()
    # Get the last entry from the database
    cur.execute("SELECT date, weight FROM weight_records ORDER BY date DESC LIMIT 1")
    last_entry = cur.fetchone()
    
    if last_entry:
        last_entry_date = datetime.strptime(last_entry[0], '%Y-%m-%d').date()
        last_weight = last_entry[1]
    else:
        last_entry_date = new_date - timedelta(days=1)
        last_weight = new_weight
    
    # Calculate the difference in days between the last entry and the new date
    delta = (new_date - last_entry_date).days
    
    # If there are missing dates, fill them with the last recorded weight
    for i in range(1, delta):
        missing_date = last_entry_date + timedelta(days=i)
        cur.execute("INSERT INTO weight_records (date, weight) VALUES (?, ?)", 
                    (missing_date.strftime('%Y-%m-%d'), last_weight))
    
    # Update the weight for the new date
    cur.execute("INSERT OR REPLACE INTO weight_records (date, weight) VALUES (?, ?)", 
                (new_date.strftime('%Y-%m-%d'), new_weight))
    
    conn.commit()
    
    # Return the number of days filled to inform the user
    return max(0, delta - 1)


# Streamlit interface for weight entry
def main():
    with get_connection() as conn:
        st.title('Eric Mei BodyWeight Magic Tool!')

        # Input for today's weight
        today = datetime.now().date()
        last_weight = get_last_recorded_weight(conn)
        st.subheader(f"Record Weight for Today: {today}")
        input_weight = st.number_input('Enter your weight (kg):', min_value=70.0, max_value=95.0, value=last_weight, step=0.1)

        # Button to record the weight
        if st.button('Record Weight'):
            # Fill missing dates with the last recorded weight and update today's weight
            days_filled = fill_missing_dates(conn, today, input_weight)
            
            if days_filled:
                st.info(f"Weight data for {days_filled} day(s) filled automatically with the last recorded weight.")
            st.success('Weight recorded successfully!')
        
        # Display the existing records
        st.subheader('Weight Records for Last 7 Days')
        df = pd.read_sql_query("SELECT * FROM weight_records Order BY date DESC LIMIT 7", conn)
        df.reset_index(drop=True, inplace=True)
        st.dataframe(df)

if __name__ == "__main__":
    main()
