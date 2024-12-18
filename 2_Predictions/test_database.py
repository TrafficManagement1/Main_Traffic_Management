import sqlite3

try:
    conn = sqlite3.connect('vehicle_data.db')
    cursor = conn.cursor()

    # Update the day_of_week column with correct values based on the date
    updates = [
        (1, 'Sunday'),
        (2, 'Monday'),
        (3, 'Tuesday'),
        (4, 'Wednesday'),
        (5, 'Thursday'),
        (6, 'Friday'),
        (7, 'Saturday'),
    ]
    for id_, day in updates:
        cursor.execute("UPDATE daily_vehicle_counts SET day_of_week = ? WHERE id = ?", (day, id_))

    conn.commit()
    print("day_of_week column updated successfully.")

    conn.close()
except Exception as e:
    print("Error:", e)
