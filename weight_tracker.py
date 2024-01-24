import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

# Function to connect to the database
class DBConnection:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

@st.cache_resource
def get_connection():
    return DBConnection('my_weight_tracker.db')

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
    conn = get_connection()
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
    styled_df = df.style.hide_index()
    st.write(styled_df)

if __name__ == "__main__":
    main()
