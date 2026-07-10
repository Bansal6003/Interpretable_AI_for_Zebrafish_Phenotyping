from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import time

# TEMPORARY workaround - use only if you're aware of the risks
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLO11 model
model = YOLO(r"C:\Users\Pushkar Bansal\runs\detect\train2\weights\last.pt")

# Open the video file
video_path = r"D:\Behavioral genetics_V1\Metamorph_scans\Maram tracker\Maram_frames\DHEA 3_Trim.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Resolution: {width}x{height}")

# Store the track history and data for each object
track_history = defaultdict(lambda: [])
tracking_data = defaultdict(lambda: {'frames': [], 'positions': [], 'distances': [], 'velocities': [], 'timestamps': []})

# Define a scaling factor (pixels to real-world units if available)
scaling_factor = 1.0  # pixels to your preferred unit (e.g., cm, meters)

frame_number = 0
start_time = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        frame_number += 1
        current_time = time.time() - start_time
        
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True)[0]
        
        # Get the boxes and track IDs
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            
            # Visualize the result on the frame
            frame = result.plot()
            
            # Process each tracked object
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                
                # Add current position to track history
                track.append((float(x), float(y)))  # x, y center point
                
                # Store current position and frame number in tracking data
                tracking_data[track_id]['frames'].append(frame_number)
                tracking_data[track_id]['positions'].append((float(x), float(y)))
                tracking_data[track_id]['timestamps'].append(current_time)
                
                # Calculate distance and velocity if we have at least two positions
                if len(track) > 1:
                    # Calculate distance between current and previous position
                    prev_x, prev_y = track[-2]
                    distance = float(np.sqrt((x - prev_x)**2 + (y - prev_y)**2) * scaling_factor)
                    
                    # Calculate velocity (distance / time between frames)
                    time_diff = 1.0 / fps  # time between frames in seconds
                    velocity = float(distance / time_diff)  # units per second
                    
                    # Store distance and velocity
                    tracking_data[track_id]['distances'].append(distance)
                    tracking_data[track_id]['velocities'].append(velocity)
                else:
                    # For the first position, set distance and velocity to 0
                    tracking_data[track_id]['distances'].append(0.0)
                    tracking_data[track_id]['velocities'].append(0.0)
                
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                
                # Display distance and velocity on the frame
                if len(tracking_data[track_id]['distances']) > 0:
                    dist_text = f"Dist: {tracking_data[track_id]['distances'][-1]:.2f} px"
                    vel_text = f"Vel: {tracking_data[track_id]['velocities'][-1]:.2f} px/s"
                    cv2.putText(frame, dist_text, (int(x) + 10, int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, vel_text, (int(x) + 10, int(y) + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Process the tracking data and create a DataFrame for Excel export
data_rows = []

for track_id, data in tracking_data.items():
    for i in range(len(data['frames'])):
        # Ensure we don't go out of bounds for distances and velocities
        distance = float(data['distances'][i]) if i < len(data['distances']) else 0.0
        velocity = float(data['velocities'][i]) if i < len(data['velocities']) else 0.0
        
        # Create a row for this data point
        row = {
            'track_id': int(track_id),
            'frame': int(data['frames'][i]),
            'timestamp': float(data['timestamps'][i]),
            'x_position': float(data['positions'][i][0]),
            'y_position': float(data['positions'][i][1]),
            'distance': distance,
            'velocity': velocity
        }
        data_rows.append(row)

# Create a DataFrame from the collected data
df = pd.DataFrame(data_rows)

# Ensure distance column is properly converted to numeric type
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

# Calculate cumulative distance for each track_id
# Calculate cumulative distance by grouping first, then applying cumsum to each group
df['cumulative_distance'] = df.groupby('track_id')['distance'].transform('cumsum')

# Define the output path for the Excel file
output_path = r"C:\Users\Pushkar Bansal\Desktop\tracking_data.xlsx"

# Create a Excel writer object
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Write complete data to the first sheet
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    # Create summary statistics for each track_id
    summary_data = []
    for track_id, group in df.groupby('track_id'):
        summary = {
            'track_id': track_id,
            'total_frames': len(group),
            'first_appearance': group['frame'].min(),
            'last_appearance': group['frame'].max(),
            'total_distance': group['cumulative_distance'].max(),
            'avg_velocity': group['velocity'].mean(),
            'max_velocity': group['velocity'].max()
        }
        summary_data.append(summary)
    
    # Create a summary DataFrame and write to second sheet
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"Tracking data has been saved to {output_path}")