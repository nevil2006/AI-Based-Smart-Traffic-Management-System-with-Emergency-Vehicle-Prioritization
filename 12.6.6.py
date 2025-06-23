import cv2
import numpy as np
import json
import time
from ultralytics import YOLO

model = YOLO(r"C:\Users\nevil\Desktop\vechile detection\best (6).pt")

video_paths = [
    r"C:\Users\nevil\Desktop\vechile detection\1.mp4", 
    r"C:\Users\nevil\Desktop\vechile detection\2.mp4",
    r"C:\Users\nevil\Desktop\vechile detection\3.mp4",
    r"C:\Users\nevil\Desktop\vechile detection\4.mp4"
]

caps = [cv2.VideoCapture(path) for path in video_paths]
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_paths[idx]}")
        exit()

polygon_points_list = [
    np.array([(467, 190), (467, 195), (464, 330), (766, 330), (578, 190)], np.int32).reshape((-1, 1, 2)),
    np.array([(460, 195), (466, 197), (456, 330), (766, 330), (581, 201)], np.int32).reshape((-1, 1, 2)),
    np.array([(462, 200), (463, 198), (463, 335), (798,335), (574, 193)], np.int32).reshape((-1, 1, 2)),
    np.array([(470, 202), (470, 202), (467, 320), (789, 320), (584, 203)], np.int32).reshape((-1, 1, 2))
]

lane_weights = {
    "bike": 1, "motorbike": 1, "bicycle": 1,
    "car": 2, "van": 2, "auto": 2,
    "bus": 5, "truck": 5
}

def calculate_green_time(total_weight):
    if total_weight < 10:
        return 15
    elif total_weight >= 90:
        return 90
    else:
        return min(90, 15 + ((total_weight - 10) // 10 + 1) * 5)

vehicle_counts_list = [{} for _ in range(4)]
vehicles_inside_list = [{} for _ in range(4)]

cv2.namedWindow("Traffic Surveillance", cv2.WINDOW_NORMAL)

# --- Initial Yellow Phase (15s) for All Lanes ---
print("\nInitial Yellow Phase for all lanes (15s)")
yellow_start = time.time()
while time.time() - yellow_start < 15:
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (960, 540))
        cv2.putText(frame, f"Lane {idx + 1} - Yellow", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        caps[idx].set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
    time.sleep(1)

# --- Signal Phase Loop ---
current_lane = 0
while True:
    # --- Calculate green time for current lane ---
    vehicle_counts = vehicle_counts_list[current_lane]
    total_weight = sum(vehicle_counts.get(v, 0) * lane_weights.get(v, 1) for v in vehicle_counts)
    green_time = calculate_green_time(total_weight)

    print(f"\nLane {current_lane+1} GREEN for {green_time}s")
    green_start = time.time()
    while time.time() - green_start < green_time:
        frames = []
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (960, 540))

            polygon_points = polygon_points_list[idx]
            results = model.track(frame, persist=True, device='0', iou=0.45, conf=0.4)
            vehicle_counts = vehicle_counts_list[idx]
            vehicles_inside = vehicles_inside_list[idx]
            current_vehicles = {}

            if results and hasattr(results[0], "boxes"):
                boxes = results[0].boxes
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if boxes.id is not None:
                        oid = int(boxes.id[i].item())
                    else:
                        continue
                    cls_id = int(boxes.cls[i].item())
                    cls_name = model.names.get(cls_id, "unknown").lower()
                    inside = cv2.pointPolygonTest(polygon_points, (cx, cy), False) >= 0
                    if inside:
                        current_vehicles[oid] = cls_name
                        if oid not in vehicles_inside:
                            vehicles_inside[oid] = cls_name
                            vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = cls_name
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            exited = set(vehicles_inside.keys()) - set(current_vehicles.keys())
            for oid in exited:
                cls = vehicles_inside[oid]
                vehicle_counts[cls] = max(0, vehicle_counts.get(cls, 0) - 1)
                del vehicles_inside[oid]

            cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

            # Status text
            if idx == current_lane:
                status = f"Lane {idx + 1} - Green ({green_time - int(time.time() - green_start)}s)"
                color = (0, 255, 0)
            elif (idx == (current_lane + 1) % 4):
                status = f"Lane {idx + 1} - Red ({green_time - int(time.time() - green_start)}s)"
                color = (0, 0, 255)
            else:
                status = f"Lane {idx + 1} - Red"
                color = (0, 0, 255)

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            frames.append(frame)

        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        full_frame = np.vstack((top, bottom))
        cv2.imshow("Traffic Surveillance", full_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()
            exit()

    # --- Yellow Phase for current and next lane ---
    print(f"Lane {current_lane + 1} & Lane {(current_lane + 1) % 4 + 1}  for 5s")
    yellow_start = time.time()
    while time.time() - yellow_start < 5:
        frames = []
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (960, 540))
            if idx == current_lane or idx == (current_lane + 1) % 4:
                text = f"Lane {idx + 1} - Yellow"
                color = (0, 255, 255)
            else:
                text = f"Lane {idx + 1} - Red"
                color = (0, 0, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            frames.append(frame)

        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        full_frame = np.vstack((top, bottom))
        cv2.imshow("Traffic Surveillance", full_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()
            exit()

    current_lane = (current_lane+1)%4
    