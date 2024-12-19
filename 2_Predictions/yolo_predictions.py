import cv2
import numpy as np
import yaml
import sqlite3
from yaml.loader import SafeLoader
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import math


class SORTTracker:
    def __init__(self, max_age=40, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age  # Increase age to maintain tracks longer
        self.min_hits = min_hits  # Increase hits to confirm detection
        self.iou_threshold = iou_threshold  # Fine-tune IoU for more accurate matches
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        if len(self.trackers) == 0:
            self.trackers = [TrackerObject(det) for det in detections]
            return [(track.id, idx, track.box) for idx, track in enumerate(self.trackers)]

        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self.iou(trk.box, det)

        # Hungarian algorithm to match detections and trackers based on IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = []
        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                unmatched_trackers.remove(r)
                unmatched_detections.remove(c)

        # Update matched trackers with new detection info
        for t, d in matches:
            self.trackers[t].update(detections[d])
            self.trackers[t].det_idx = d

        # Initialize new trackers for unmatched detections
        for i in unmatched_detections:
            new_tracker = TrackerObject(detections[i])
            new_tracker.det_idx = i
            self.trackers.append(new_tracker)

        # Filter trackers based on min_hits and max_age
        self.trackers = [t for t in self.trackers if t.hit_streak >= self.min_hits or t.time_since_update < self.max_age]
        return [(track.id, track.det_idx, track.box) for track in self.trackers if track.hit_streak >= self.min_hits]

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea) if interArea > 0 else 0

class TrackerObject:
    _id_count = 0
    def __init__(self, box):
        self.id = TrackerObject._id_count
        TrackerObject._id_count += 1
        self.box = box
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.det_idx = None
        self.last_position = box[:2]  # Store initial position (x1, y1)
    
    def update(self, box):
        self.box = box
        self.last_position = box[:2]  # Update last position
        self.time_since_update = 0
        self.hit_streak += 1



class YOLO_Pred:
    
    def __init__(self, onnx_model, data_yaml):
        # Load data from YAML file, including configurable confidence threshold
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        self.confidence_threshold = data_yaml.get('confidence_threshold', 0.3)  # Default to 0.3 if not specified
        self.vehicle_count = {label: 0 for label in self.labels}
        self.total_vehicle_count = {label: 0 for label in self.labels}
        self.counted_vehicles = set()  # Initialize counted_vehicles set here
        self.time_based_counts = {}
        self.historical_counts = []
        self.vehicle_colors = {
            'car': (0, 255, 0),         # Green
            'motorcycle': (255, 0, 0),   # Blue
            'truck': (0, 0, 255),       # Red
            'bus': (255, 255, 0),       # Cyan
            'jeep': (255, 0, 255),      # Magenta
            'tricycle': (0, 255, 255)   # Yellow
        }
        
        # Load the YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Drawing line configuration
        self.line_y = 500
        self.line_thickness = 2
        self.line_color = (0, 255, 0)
        self.congestion_level = "Light Traffic"
        # Update congestion level after counting vehicles
        self.calculate_congestion()
        self.vehicle_count_last_minute = 0
        
        # Initialize SORT tracker
        self.tracker = SORTTracker()

    @staticmethod
    def init_historical_db():
        conn = sqlite3.connect('vehicle_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_congestion (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                congestion_level TEXT,
                car_count INTEGER,
                truck_count INTEGER,
                motorcycle_count INTEGER,
                bus_count INTEGER,
                jeep_count INTEGER,
                tricycle_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()


    def log_historical_data(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect('vehicle_data.db')
        cursor = conn.cursor()
        
        # Insert the current snapshot of vehicle counts and congestion level
        cursor.execute('''
            INSERT INTO historical_congestion (
                timestamp, congestion_level, car_count, truck_count, motorcycle_count, 
                bus_count, jeep_count, tricycle_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_time,
            self.congestion_level,
            self.total_vehicle_count.get('car', 0),
            self.total_vehicle_count.get('truck', 0),
            self.total_vehicle_count.get('motorcycle', 0),
            self.total_vehicle_count.get('bus', 0),
            self.total_vehicle_count.get('jeep', 0),
            self.total_vehicle_count.get('tricycle', 0)
        ))
        
        # Delete records older than 30 days
        cursor.execute("DELETE FROM historical_congestion WHERE timestamp < datetime('now', '-30 days')")
        
        conn.commit()
        conn.close()
        print("Historical data logged and old data cleaned.")

    def log_daily_counts_to_db(self):
        """Logs the aggregated daily vehicle counts to the SQLite database."""
        try:
            conn = sqlite3.connect('vehicle_data.db')
            cursor = conn.cursor()

            today_date = datetime.now().strftime("%Y-%m-%d")

            # Check if today's record already exists
            cursor.execute(
                "SELECT car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count FROM daily_vehicle_counts WHERE date = ?",
                (today_date,)
            )
            current_data = cursor.fetchone()

            # Aggregate current counts
            if current_data:
                updated_data = tuple(
                    current_data[i] + self.total_vehicle_count.get(vehicle, 0)
                    for i, vehicle in enumerate(['car', 'truck', 'motorcycle', 'bus', 'jeep', 'tricycle'])
                )
                cursor.execute(
                    '''
                    UPDATE daily_vehicle_counts
                    SET car_count = ?, truck_count = ?, motorcycle_count = ?, bus_count = ?, jeep_count = ?, tricycle_count = ?
                    WHERE date = ?
                    ''',
                    (*updated_data, today_date)
                )
            else:
                # Insert a new record if none exists for today
                cursor.execute(
                    '''
                    INSERT INTO daily_vehicle_counts (date, car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        today_date,
                        self.total_vehicle_count.get('car', 0),
                        self.total_vehicle_count.get('truck', 0),
                        self.total_vehicle_count.get('motorcycle', 0),
                        self.total_vehicle_count.get('bus', 0),
                        self.total_vehicle_count.get('jeep', 0),
                        self.total_vehicle_count.get('tricycle', 0),
                    )
                )

            conn.commit()
            print(f"[INFO] Daily counts logged successfully for {today_date}.")
        except Exception as e:
            print(f"[ERROR] Error logging daily counts: {e}")
        finally:
            conn.close()

    def calculate_congestion(self):
        total_vehicle_count = sum(self.total_vehicle_count.values())

        if total_vehicle_count < 300:
            self.congestion_level = "Low Volume"
        elif 30 <= total_vehicle_count <= 500:
            self.congestion_level = "Mid Volume"
        else:
            self.congestion_level = "High Volume"
        # Reset count for next interval
        self.vehicle_count_last_minute = 0

    def update_vehicle_count(self, new_vehicles):
        """Update vehicle count for each interval and determine congestion level."""
        self.vehicle_count_last_minute += new_vehicles
        self.calculate_congestion()  # Calculate congestion based on updated count

    def draw_dashed_line(self, image, start, end, color, thickness=2, gap=10):
        """Draws a dashed line on the image from start to end points."""
        start_x = max(0, start[0])
        start_y = max(0, start[1])
        end_x = min(image.shape[1] - 1, end[0])
        end_y = min(image.shape[0] - 1, end[1])

        line_length = np.linalg.norm(np.array([end_x, end_y]) - np.array([start_x, start_y]))
        for i in range(0, int(line_length), gap * 2):
            start_dash = (int(start_x + (end_x - start_x) * i / line_length),
                          int(start_y + (end_y - start_y) * i / line_length))
            end_dash = (int(start_x + (end_x - start_x) * (i + gap) / line_length),
                        int(start_y + (end_y - start_y) * (i + gap) / line_length))
            cv2.line(image, start_dash, end_dash, color, thickness)

    def log_counts_to_db(self):
        """Logs the current vehicle counts to the SQLite database."""
        try:
            # Establish a connection to the database
            conn = sqlite3.connect('vehicle_data.db')
            cursor = conn.cursor()
            
            # Get the current time for the record
            current_time = datetime.now().strftime("%Y-%m-%d %H:00")
            
            # Loop through each vehicle type and insert into the database if count > 0
            for vehicle_type, count in self.total_vehicle_count.items():
                if count > 0:  # Only log non-zero counts
                    # Debug print before insertion
                    print(f"Inserting into DB: Time: {current_time}, Vehicle Type: {vehicle_type}, Count: {count}")
                    cursor.execute(
                        "INSERT INTO vehicle_counts (timestamp, vehicle_type, count) VALUES (?, ?, ?)",
                        (current_time, vehicle_type, count)
                    )
            
            # Commit changes and close the connection
            conn.commit()
        except Exception as e:
            print(f"Error logging counts to DB: {e}")
        finally:
            if conn:
                conn.close()
                
                
    def update_hourly_counts(self):
        """Update hourly vehicle counts in the database."""
        try:
            conn = sqlite3.connect('Notes/vehicle_data.db')
            cursor = conn.cursor()

            current_hour = datetime.now().strftime("%Y-%m-%d %H:00:00")

            # Insert or update the hourly count
            cursor.execute('''
                INSERT INTO hourly_vehicle_counts (hour, car_count, truck_count, motorcycle_count, bus_count, jeep_count, tricycle_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hour) DO UPDATE SET
                car_count = excluded.car_count,
                truck_count = excluded.truck_count,
                motorcycle_count = excluded.motorcycle_count,
                bus_count = excluded.bus_count,
                jeep_count = excluded.jeep_count,
                tricycle_count = excluded.tricycle_count
            ''', (
                current_hour,
                self.total_vehicle_count.get('car', 0),
                self.total_vehicle_count.get('truck', 0),
                self.total_vehicle_count.get('motorcycle', 0),
                self.total_vehicle_count.get('bus', 0),
                self.total_vehicle_count.get('jeep', 0),
                self.total_vehicle_count.get('tricycle', 0),
            ))

            conn.commit()
            print(f"Hourly counts updated for {current_hour}.")
        except Exception as e:
            print(f"Error updating hourly counts: {e}")
        finally:
            conn.close()


    def predictions(self, image):
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        self.vehicle_count = {label: 0 for label in self.labels}

        # Collecting detections and applying score threshold
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > self.confidence_threshold:  # Use configurable threshold
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > self.confidence_threshold:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Bounding box format to (x1, y1, x2, y2) for drawing
                    box = np.array([left, top, left + width, top + height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Apply Non-Maximum Suppression to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.confidence_threshold, nms_threshold=0.2)


        
        if len(indices) > 0:
            indices = indices.flatten()  # Ensure indices is a flat list

        # Only keep boxes, confidences, and classes selected by NMS
        nms_boxes = [boxes[i] for i in indices]
        nms_classes = [classes[i] for i in indices]

        # Track objects using SORT and draw boxes
        tracked_objects = self.tracker.update(nms_boxes)

        for obj_id, det_idx, box in tracked_objects:
            if box is not None and det_idx is not None and 0 <= det_idx < len(nms_classes):  # Ensure both box and det_idx are valid
                x1, y1, x2, y2 = box  # Bounding box format
                class_name = self.labels[nms_classes[det_idx]]

                color = self.vehicle_colors.get(class_name, (0, 255, 0))  # Default color if not found

                # Get the tracked object and its last position
                tracker_object = self.tracker.trackers[obj_id]
                last_position = tracker_object.last_position
                current_position = (x1, y1)

                # Calculate the distance moved from the last position
                distance_moved = calculate_distance(last_position, current_position)

                if self.line_y - 80 < y2 < self.line_y + 80:
                    if distance_moved < 300:
                        if obj_id not in self.counted_vehicles:
                            # All conditions met, count the vehicle
                            self.total_vehicle_count[class_name] += 1
                            self.counted_vehicles.add(obj_id)
                            # Update hourly counts in the database
                            self.update_hourly_counts()  # <<< Add here
                        else:
                            print(f"Skipping object ID {obj_id}: already counted.")
                    else:
                        print(f"Skipping object ID {obj_id}: distance moved ({distance_moved}) too large.")
                else:
                    print(f"Skipping object ID {obj_id}: not in line-crossing range (y2={y2}).")

        for obj_id, det_idx, box in tracked_objects:
            if box is not None and det_idx is not None and 0 <= det_idx < len(nms_classes):
                x1, y1, x2, y2 = box
                class_name = self.labels[nms_classes[det_idx]]

                # Updated code for face blurring
                if class_name == 'face':
                    face_region = image[y1:y2, x1:x2]
                    if face_region.size > 0:
                        blurred_face = cv2.GaussianBlur(face_region, (35, 35), 30)
                        image[y1:y2, x1:x2] = blurred_face
                else:
                    color = self.vehicle_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f'ID {obj_id} - {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)

            else:
                # Print a more informative warning to help debug why this block is not executing
                print(f"Skipping object ID {obj_id}: det_idx={det_idx}, box={box}")

        # Draw the dashed line on the frame
        self.draw_dashed_line(image, start=(0, self.line_y), end=(image.shape[1], self.line_y), color=(0, 255, 0), thickness=2)

        return image

    
def calculate_distance(pos1, pos2):
    if pos1 is None or pos2 is None:
        return float('inf')  # Infinite distance if one of the positions is undefined
    return np.linalg.norm(np.array(pos2) - np.array(pos1))
