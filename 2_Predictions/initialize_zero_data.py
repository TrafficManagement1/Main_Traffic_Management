import sqlite3

def initialize_zero_data():
    conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/Notes/2_Predictions/vehicle_data.db')
    cursor = conn.cursor()

    # Insert zero counts for Sun to Sat
    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO daily_vehicle_counts (date, car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [
            ('2024-12-01', 0, 0, 0, 0, 0, 0),  # Sun
            ('2024-12-02', 0, 0, 0, 0, 0, 0),  # Mon
            ('2024-12-03', 0, 0, 0, 0, 0, 0),  # Tue
            ('2024-12-04', 0, 0, 0, 0, 0, 0),  # Wed
            ('2024-12-05', 0, 0, 0, 0, 0, 0),  # Thu
            ('2024-12-06', 0, 0, 0, 0, 0, 0),  # Fri
            ('2024-12-07', 0, 0, 0, 0, 0, 0)   # Sat
        ])
        conn.commit()
        print("[INFO] Initialized zero data for the week.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize data: {e}")
    finally:
        conn.close()

def create_daily_counts_table():
    """
    Create the daily_vehicle_counts table if it does not already exist.
    """
    # Connect to the database
    conn = sqlite3.connect('vehicle_data.db')  # Replace with the correct path to your database
    cursor = conn.cursor()

    # SQL command to create the table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_vehicle_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_of_week TEXT NOT NULL,
            total_count INTEGER NOT NULL,
            date DATE UNIQUE NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    print("Table 'daily_vehicle_counts' created or already exists.")

if __name__ == "__main__":
    initialize_zero_data()
    create_daily_counts_table()
