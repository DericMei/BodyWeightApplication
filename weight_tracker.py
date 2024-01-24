import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

# Function to connect to the database
@st.cache(hash_funcs={sqlite3.Connection: id})
def get_connection():
    return sqlite3.connect('my_weight_tracker.db')

# Function to get the last recored weight
def get_last_recorded_weight(conn):
    query = "SELECT weight FROM weight_records ORDER BY date DESC LIMIT 1"
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()
    return result[0] if result else None

# Function to insert or update weight data
def upsert_weight_data(conn, date, weight):
    cur = conn.cursor()
    # Check if entry for today already exists
    cur.execute("SELECT weight FROM weight_records WHERE date = ?", (date,))
    if cur.fetchone():
        # Update the record if entry for today exists
        cur.execute("UPDATE weight_records SET weight = ? WHERE date = ?", (weight, date))
    else:
        # Insert a new record if no entry for today
        cur.execute("INSERT INTO weight_records (date, weight) VALUES (?, ?)", (date, weight))
    conn.commit()

# Streamlit interface for weight entry
def main():
    conn = get_connection()
    st.title('Daily Weight Tracker')

    # Input for today's weight
    today = datetime.now().date()
    last_weight = get_last_recorded_weight(conn)
    
    st.subheader(f"Record Weight for Today: {today}")
    input_weight = st.number_input('Enter your weight (kg):', value=last_weight)

    # Button to record the weight
    if st.button('Record Weight'):
        upsert_weight_data(conn, today, input_weight)
        st.success('Weight recorded successfully!')
    
    # Display the existing records
    st.subheader('Weight Records')
    df = pd.read_sql_query("SELECT * FROM weight_records", conn)
    st.write(df)

if __name__ == "__main__":
    main()
