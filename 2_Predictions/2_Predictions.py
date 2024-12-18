from flask import Flask, jsonify, send_file, Response
import cv2
from yolo_predictions import sqlite3
from yolo_predictions import datetime
from yolo_predictions import YOLO_Pred
from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3

app = Flask(__name__)

# Initialize YOLO model
yolo = YOLO_Pred(
    'C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/Model16/weights/best.onnx',
    'C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/data.yaml'
)

# Initialize scheduler
scheduler = BackgroundScheduler()

# Schedule daily aggregation at midnight
scheduler.add_job(yolo.log_daily_counts_to_db, 'cron', hour=0, minute=0)

# Open video file
video_path = 'https://twitch.tv/tarfic01'
cap = cv2.VideoCapture(video_path)

def log_counts_to_db(self):
    conn = sqlite3.connect('vehicle_data.db')
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:00")
    for vehicle_type, count in self.total_vehicle_count.items():
        if count > 0:  # Only log non-zero counts
            cursor.execute(
                "INSERT INTO vehicle_counts (timestamp, vehicle_type, count) VALUES (?, ?, ?)",
                (current_time, vehicle_type, count)
            )
    conn.commit()
    conn.close()

# Initialize the database if it doesn't exist
def initialize_database():
    conn = sqlite3.connect('vehicle_data.db')
    cursor = conn.cursor()

    # Create vehicle_counts table (for hourly data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_type TEXT,
            count INTEGER
        )
    ''')

    # Create daily_vehicle_counts table (for daily data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_vehicle_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            car_count INTEGER DEFAULT 0,
            truck_count INTEGER DEFAULT 0,
            motorcycle_count INTEGER DEFAULT 0,
            bus_count INTEGER DEFAULT 0,
            jeep_count INTEGER DEFAULT 0,
            tricycle_count INTEGER DEFAULT 0
        )
    ''')

    # Optional: Add other table initializations here if needed

    conn.commit()
    conn.close()
    print("[INFO] Database initialized successfully.")

initialize_database()

# Set up the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(yolo.log_historical_data, 'interval', hours=1)


# Video frame generator
frame_count = 0

def generate_frames():
    global frame_count
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = yolo.predictions(frame)
        frame_count += 1

        # Log counts every 30 frames (~1 second for a 30 FPS video)
        if frame_count % 30 == 0:
            yolo.log_counts_to_db()

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
        return send_file(r"C:\Users\moise\OneDrive\Desktop\Main_Traffic__Management\index.html")

@app.route('/video_feed')
def video_feed():
    # Serve the processed video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Real-time counts for the dashboard
# Store the last interval counts in memory
last_interval_counts = {'12-2am': 0, '2-4am': 0, '4-6am': 0, '6-8am': 0, 
                        '8-10am': 0, '10-12nn': 0, '12-2pm': 0, '2-4pm': 0, 
                        '4-6pm': 0, '6-8pm': 0, '8-10pm': 0, '10-12mn': 0}

@app.route('/get_report_data')
def get_report_data():
    try:
        # Fetch the accurate counts from self.total_vehicle_count
        current_counts = {
            'car': yolo.total_vehicle_count.get('car', 0),
            'motorcycle': yolo.total_vehicle_count.get('motorcycle', 0),
            'truck': yolo.total_vehicle_count.get('truck', 0),
            'bus': yolo.total_vehicle_count.get('bus', 0),
            'jeep': yolo.total_vehicle_count.get('jeep', 0),
            'tricycle': yolo.total_vehicle_count.get('tricycle', 0),
        }

        # Connect to the database
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/vehicle_data.db')
        cursor = conn.cursor()

        # Get the current date
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Update total counts for each vehicle type
        cursor.execute(
            """
            INSERT INTO daily_graph (date, car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                car_count = ?,
                truck_count = ?,
                motorcycle_count = ?,
                bus_count = ?,
                jeep_count = ?,
                tricycle_count = ?;
            """,
            (
                today_date,
                current_counts['car'],
                current_counts['truck'],
                current_counts['motorcycle'],
                current_counts['bus'],
                current_counts['jeep'],
                current_counts['tricycle'],
                current_counts['car'],
                current_counts['truck'],
                current_counts['motorcycle'],
                current_counts['bus'],
                current_counts['jeep'],
                current_counts['tricycle'],
            )
        )

        # Determine the current 2-hour interval
        hour = datetime.now().hour
        interval_map = {
            range(0, 2): "12-2am", range(2, 4): "2-4am", range(4, 6): "4-6am",
            range(6, 8): "6-8am", range(8, 10): "8-10am", range(10, 12): "10-12nn",
            range(12, 14): "12-2pm", range(14, 16): "2-4pm", range(16, 18): "4-6pm",
            range(18, 20): "6-8pm", range(20, 22): "8-10pm", range(22, 24): "10-12mn"
        }
        current_interval = next(val for key, val in interval_map.items() if hour in key)

        # Calculate the new counts only for the current interval
        new_interval_count = sum(current_counts.values()) - last_interval_counts[current_interval]
        last_interval_counts[current_interval] += new_interval_count

        # Update the respective time interval column with only the new count
        cursor.execute(f"""
            UPDATE daily_graph
            SET "{current_interval}" = COALESCE("{current_interval}", 0) + ?
            WHERE date = ?;
        """, (new_interval_count, today_date))

        # Commit and close
        conn.commit()
        conn.close()

        print(f"[INFO] Counts updated for {today_date} and interval '{current_interval}' with new count: {new_interval_count}")
        return jsonify(current_counts)

    except Exception as e:
        print(f"[ERROR] Failed to log accurate counts to database: {e}")
        return jsonify({"error": str(e)}), 500



# Aggregated daily data for descriptive analytics
@app.route('/get_daily_data', methods=['GET'])
def get_daily_data():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/vehicle_data.db')
        cursor = conn.cursor()

        # Query to get all dates and the total counts (sum of all vehicle types)
        cursor.execute('''
            SELECT date, 
                   (car_count + truck_count + motorcycle_count + bus_count + jeep_count + tricycle_count) AS total_count
            FROM daily_graph
            ORDER BY date ASC;
        ''')

        # Fetch all rows from the query
        rows = cursor.fetchall()

        # Format data into a list of dictionaries
        data = [{"date": row[0], "count": row[1]} for row in rows]

        # Close the database connection
        conn.close()

        # Return the data as JSON
        return jsonify({"data": data})

    except Exception as e:
        print(f"[ERROR] Failed to fetch daily data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_hourly_data', methods=['GET'])
def get_hourly_data():
    try:
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/vehicle_data.db')
        cursor = conn.cursor()

        # Fetch cumulative counts for all time intervals
        today_date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(f"""
            SELECT "12-2am", "2-4am", "4-6am", "6-8am", "8-10am", "10-12nn",
                   "12-2pm", "2-4pm", "4-6pm", "6-8pm", "8-10pm", "10-12mn"
            FROM daily_graph WHERE date = ?;
        """, (today_date,))

        row = cursor.fetchone()
        hourly_data = [value if value else 0 for value in row] if row else [0] * 12

        conn.close()
        return jsonify({"counts": hourly_data})
    except Exception as e:
        print(f"Error fetching hourly data: {e}")
        return jsonify({"error": str(e)}), 500




@app.route('/get_descriptive_data')
def get_descriptive_data():
    return jsonify(yolo.time_based_counts)
    
@app.route('/get_congestion_level', methods=['GET'])
def get_congestion_level():
    yolo.calculate_congestion()  # Calculate the current congestion level
    congestion_level = yolo.congestion_level
    print(f'Current congestion level set to: {congestion_level}')  # Debug output to verify
    return jsonify({'congestion_level': congestion_level})

@app.route('/get_historical_congestion_data')
def get_historical_congestion_data():
    conn = sqlite3.connect('vehicle_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, congestion_level, car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count FROM historical_congestion ORDER BY timestamp')
    data = cursor.fetchall()
    conn.close()
    
    historical_data = [
        {
            "timestamp": row[0],
            "congestion_level": row[1],
            "car_count": row[2],
            "truck_count": row[3],
            "motorcycle_count": row[4],
            "bus_count": row[5],
            "jeep_count": row[6],
            "tricycle_count": row[7]
        } for row in data
    ]
    
    return jsonify(historical_data)


@app.route('/get_historical_data')
def get_historical_data():
    try:
        conn = sqlite3.connect('vehicle_data.db')
        cursor = conn.cursor()
        # Retrieve data for the last 7 days
        cursor.execute('''
            SELECT timestamp, vehicle_type, SUM(count) as total_count
            FROM vehicle_counts
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY timestamp, vehicle_type
        ''')
        data = cursor.fetchall()
    except Exception as e:
        print("Database error:", e)
        return jsonify({"error": "Database query failed"}), 500
    finally:
        conn.close()

    # Format data as needed for frontend
    historical_data = {}
    for row in data:
        timestamp, vehicle_type, count = row
        if timestamp not in historical_data:
            historical_data[timestamp] = {}
        historical_data[timestamp][vehicle_type] = count
    return jsonify(historical_data)

@app.route('/manual_log_daily_counts')
def manual_log_daily_counts():
    yolo.log_daily_counts_to_db()
    return jsonify({"status": "Manual daily counts logging triggered"}) 

@app.route('/get_predictive_data', methods=['GET'])
def get_predictive_data():
    try:
        conn = sqlite3.connect('vehicle_data.db')
        cursor = conn.cursor()

        # Aggregate historical data for predictions
        cursor.execute('''
            SELECT 
                strftime('%w', timestamp) as day_of_week, -- Day of week (0=Sunday)
                CAST(strftime('%H', timestamp) AS INTEGER) as hour, -- Hour of the day
                SUM(count) as total_count
            FROM vehicle_counts
            GROUP BY day_of_week, hour
        ''')
        data = cursor.fetchall()
        conn.close()

        # Initialize structured data for predictions
        predictive_data = {day: [None] * 12 for day in range(7)}  # 7 days, 12 intervals
        for row in data:
            day = int(row[0])  # Day of the week
            hour = int(row[1])
            interval = hour // 2  # Map hours to 2-hour intervals
            total_count = row[2]

            # Determine traffic level
            if total_count < 30:
                level = "Light"
            elif 30 <= total_count <= 70:
                level = "Moderate"
            else:
                level = "Heavy"

            predictive_data[day][interval] = level

        return jsonify(predictive_data)
    except Exception as e:
        print("Error generating predictive data:", e)
        return jsonify({"error": "Failed to generate predictive data"}), 500

@app.route('/get_average_counts', methods=['GET'])
def get_average_counts():
    try:
        # Connect to the database
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/2_Predictions/vehicle_data.db')
        cursor = conn.cursor()

        # Correct SQL query based on your schema
        query = """
        SELECT 
            day_of_week, 
            ROUND(AVG(total_count), 2) AS avg_count
        FROM daily_vehicle_counts
        GROUP BY day_of_week
        ORDER BY date;
        """
        cursor.execute(query)
        results = cursor.fetchall()

        # Convert results into a dictionary
        average_counts = {row[0]: row[1] for row in results}

        conn.close()
        return jsonify(average_counts)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": f"Database error occurred: {e}"}), 500
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500


@app.route('/test_database', methods=['GET'])
def test_database():
    try:
        conn = sqlite3.connect('vehicle_data.db')  # Ensure this matches the location of your database
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM daily_vehicle_counts;")
        data = cursor.fetchall()
        print("Database connected successfully. Data:", data)
        conn.close()
        return jsonify({"message": "Database connected successfully", "data": data})
    except Exception as e:
        print("Database connection error:", e)
        return jsonify({"error": str(e)}), 500

from datetime import datetime

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/Notes/vehicle_data.db')
        cursor = conn.cursor()

        # Get current time interval
        now = datetime.now()
        current_hour = now.hour
        if current_hour % 2 == 0:
            start_time = f"{current_hour}:00"
            end_time = f"{current_hour + 2}:00"
        else:
            start_time = f"{current_hour - 1}:00"
            end_time = f"{current_hour + 1}:00"
        time_interval = f"{start_time}-{end_time}"
        current_date = now.strftime('%Y-%m-%d')

        # Fetch total counts for the current interval
        cursor.execute('''
            SELECT total_count
            FROM hourly_vehicle_counts
            WHERE time_interval = ? AND date = ?
        ''', (time_interval, current_date))
        result = cursor.fetchone()
        total_count = result[0] if result else 0

        recommendations = []
        if total_count >= 5:
            recommendations.append({
                "alert": "Heavy traffic ahead. Recommend you to avoid route here in this time",
                "time_interval": time_interval
            })

        conn.close()
        return jsonify(recommendations)
    except Exception as e:
        print("Error fetching recommendations:", e)
        return jsonify({"error": str(e)}), 500


def initialize_total_vehicle_counts_table():
    """Create the total_vehicle_counts table if it doesn't exist."""
    try:
        conn = sqlite3.connect('C:/Users/moise/OneDrive/Desktop/Main_Traffic__Management/Notes/vehicle_data.db')
        cursor = conn.cursor()

        # Create the table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS total_vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_type TEXT UNIQUE,
                count INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        print("[INFO] total_vehicle_counts table created successfully.")
    except Exception as e:
        print(f"[ERROR] Error creating total_vehicle_counts table: {e}")
    finally:
        conn.close()


if __name__ == '__main__':
    initialize_database()  # Initialize database and create necessary tables
    initialize_total_vehicle_counts_table()
    scheduler.start()
    app.run(debug=True)
