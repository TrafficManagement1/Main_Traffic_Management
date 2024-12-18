import sqlite3

# Connect to the database
conn = sqlite3.connect('vehicle_data.db')  # Ensure this path matches your database file
cursor = conn.cursor()

# SQL command to insert test data for yesterday
cursor.execute('''
    INSERT INTO daily_vehicle_counts (day_of_week, total_count, date)
    VALUES (4, 1029, '2024-12-05'); -- 4 = Thursday
''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Test data inserted successfully.")
