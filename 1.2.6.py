import cv2
import numpy as np
import json
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"C:\Users\nevil\Desktop\vechile detection\best (6).pt")

# Define 4 video paths
video_paths = [
    r"C:\Users\nevil\Desktop\vechile detection\1.mp4",
    r"C:\Users\nevil\Desktop\vechile detection\2.mp4",
    r"C:\Users\nevil\Desktop\vechile detection\3.mp4",
    r"C:\Users\nevil\Desktop\vechile detection\4.mp4"
]

# Open all videos
caps = [cv2.VideoCapture(path) for path in video_paths]

for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_paths[idx]}")
        exit()

# Define ROI polygons
polygon_points_list = [
    np.array([(467, 190), (467, 195), (464, 330), (766, 330), (578, 190)], np.int32).reshape((-1, 1, 2)),
    np.array([(460, 195), (466, 197), (456, 330), (766, 330), (581, 201)], np.int32).reshape((-1, 1, 2)),
    np.array([(462, 200), (463, 198), (463, 335), (798,335), (574, 193)], np.int32).reshape((-1, 1, 2)),
    np.array([(470, 202), (470, 202), (467, 320), (789, 320), (584, 203)], np.int32).reshape((-1, 1, 2))
]

# Vehicle weights
lane_weights = {
    "bike": 1, "motorbike": 1, "bicycle": 1,
    "car": 2, "van": 2,
    "bus": 3, "truck": 3
}

vehicle_counts_list = [{} for _ in range(4)]
vehicles_inside_list = [{} for _ in range(4)]

# Timing and lane control
active_lane_idx = 0
lane_change_interval = 30  # seconds
last_switch_time = time.time()

# Store final timings for report
final_lane_timings = [{} for _ in range(4)]

# Create window
cv2.namedWindow("Traffic Surveillance", cv2.WINDOW_NORMAL)

# Control flag
process_running = True

while process_running:
    frames = []
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (960, 540))
        frames.append(frame)

    if all([np.count_nonzero(frame) == 0 for frame in frames]):
        print("End of all video streams.")
        break

    processed_frames = []
    lane_timings_json = {}

    current_time = time.time()

    if current_time - last_switch_time >= lane_change_interval:
        # --- UPDATED BLOCK ---
        # Save final green time for the active lane
        if active_lane_idx < 4:
            total_lane_weight = 0
            vehicle_counts = vehicle_counts_list[active_lane_idx]

            for vtype, count in vehicle_counts.items():
                weight = lane_weights.get(vtype, 1)
                total_lane_weight += count * weight

            if total_lane_weight <= 10:
                green_time = 15
            elif 31 <= total_lane_weight <= 20:
                green_time = 20
            elif 41 <= total_lane_weight <= 25:
                green_time = 25
            elif 51 <= total_lane_weight <= 35:
                green_time = 30
            else:
                green_time = min(90, 30 + ((total_lane_weight - 60) // 10) * 5)

            green_time = max(15, min(90, green_time))

            final_lane_timings[active_lane_idx] = {
                "Red_Time_Seconds": 20,  # Assuming a fixed red time for simplicity
                "Yellow_Time_Seconds": 5,
                "Green_Time_Seconds": green_time
            }
        # --- END UPDATED ---

        active_lane_idx += 1
        last_switch_time = current_time

        if active_lane_idx >= 4:
            # All 4 lanes processed
            process_running = False
            break

    for idx, frame in enumerate(frames):
        if np.count_nonzero(frame) == 0:
            processed_frames.append(frame)
            continue

        polygon_points = polygon_points_list[idx]
        vehicle_counts = vehicle_counts_list[idx]
        vehicles_inside = vehicles_inside_list[idx]

        results = model.track(frame, persist=True, device=0, iou=0.45, conf=0.3)
        current_vehicles = {}

        if results and hasattr(results[0], "boxes"):
            boxes = results[0].boxes

            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                if boxes.id is not None:
                    object_id = int(boxes.id[i].item())
                else:
                    continue

                detected_class = int(boxes.cls[i].item())
                class_name = model.names.get(detected_class, "Unknown").lower()

                inside = cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0

                if inside:
                    current_vehicles[object_id] = class_name

                    if object_id not in vehicles_inside:
                        vehicles_inside[object_id] = class_name
                        vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

                color_map = {"car": (0, 255, 255), "bike": (0, 255, 255), "motorbike": (0, 255, 255),
                             "bus": (255, 0, 0), "van": (255, 0, 0), "truck": (255, 0, 0)}
                color = color_map.get(class_name, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, f"{class_name}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        exited = set(vehicles_inside.keys()) - set(current_vehicles.keys())
        for object_id in exited:
            class_name = vehicles_inside[object_id]
            if vehicle_counts.get(class_name, 0) > 0:
                vehicle_counts[class_name] -= 1
            del vehicles_inside[object_id]

        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

        total_lane_weight = 0
        for vtype, count in vehicle_counts.items():
            weight = lane_weights.get(vtype, 1)
            total_lane_weight += count * weight

        if idx == active_lane_idx:
            if total_lane_weight <= 30:
                green_time = 15
            elif 31 <= total_lane_weight <= 40:
                green_time = 20
            elif 41 <= total_lane_weight <= 50:
                green_time = 25
            elif 51 <= total_lane_weight <= 60:
                green_time = 30
            else:
                green_time = min(90, 30 + ((total_lane_weight - 60) // 10) * 5)

            green_time = max(15, min(90, green_time))
            red_time = 0
        else:
            green_time = 0
            red_time = 100

        lane_timings_json[f"Lane_{idx+1}"] = {
            "Green_Time_Seconds": green_time,
            "Red_Time_Seconds": red_time
        }

        cv2.rectangle(frame, (0, 0), (300, 60 + len(vehicle_counts) * 30), (0, 0, 0), thickness=cv2.FILLED)

        y_offset = 30
        for vtype, count in vehicle_counts.items():
            if count > 0:
                cv2.putText(frame, f"{vtype.capitalize()}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                y_offset += 40

        if idx == active_lane_idx:
            cv2.putText(frame, "ACTIVE", (800, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        processed_frames.append(frame)

    top = np.hstack((processed_frames[0], processed_frames[1]))
    bottom = np.hstack((processed_frames[2], processed_frames[3]))
    grid_frame = np.vstack((top, bottom))

    cv2.imshow("Traffic Surveillance", grid_frame)

    print(f"Active Lane: Lane_{active_lane_idx + 1}")
    print(json.dumps(lane_timings_json, indent=4))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End processing

# Print full final report
print("\n\n--- Final Lane Timings Report ---\n")
for idx in range(4):
    timing = final_lane_timings[idx]
    green_time = timing.get('Green_Time_Seconds', 0)
    red_time = timing.get('Red_Time_Seconds', 100)

    print(f"Lane {idx+1} Timings:")
    print(f"    Red Time: {red_time} seconds")
    print(f"    Yellow Time: 5 seconds")
    print(f"    Green Time: {green_time} seconds\n")

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()