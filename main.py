import cv2
from ultralytics import YOLO
from collections import defaultdict
from constants import *

# Initialize the YOLO model
model = YOLO(MODEL_PATH)

# Load the video file
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

# Initialize trackers and counters
class_counts = defaultdict(int)  # Store object counts by class
crossed_ids = set()  # Track IDs of objects that crossed the line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=OBJECT_CLASSES)

    if results[0].boxes.data is not None:
        # Extract detections
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw the crossing line
        cv2.line(frame, LINE_START, LINE_END, LINE_COLOR, 3)

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            class_name = model.names[class_idx]

            # Draw object details
            cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, -1)
            # Make Shadow by adding white and Black together
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_SHADOW_COLOR_BLACK, 3)  # Shadow
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_SHADOW_COLOR_WHITE, 1)  # Main text

            cv2.rectangle(frame, (x1, y1), (x2, y2), RECTANGLE_COLOR, 2)

            # FPS Counter
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {fps:.0f}", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_FPS_SCALE , FPS_COLOR, 2)

            # Update counters if the object crosses the line
            if cy > CROSSING_LINE_Y and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

        # Display class counts on the frame
        y_offset = TEXT_OFFSET_Y
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, 2)
            y_offset += TEXT_OFFSET_Y

    cv2.imshow('Tracking and Counting', frame)
    if cv2.waitKey(1) == ESCAPE_KEY:
        break

cap.release()
cv2.destroyAllWindows()
