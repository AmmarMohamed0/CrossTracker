import cv2
from ultralytics import YOLO
from collections import defaultdict

# Path to the video file for processing
video_file_path = "test/sample.mp4"
# Y-coordinate of the line for crossing detection
crossing_line_y = 430
# Dictionary to store counts of detected object classes
object_counts = defaultdict(int)  
# Set to store unique object IDs that have crossed the detection line
crossed_object_ids = set()  

# Load the YOLO object detection model
yolo_model = YOLO('yolo11s.pt')

# List of class names recognized by the model
class_names = yolo_model.names

# Open the video file for processing
video_capture = cv2.VideoCapture(video_file_path)

while video_capture.isOpened():
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Run YOLO tracking on the current frame 
    detection_results = yolo_model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])
    
    # Ensure that detected results are not empty
    if detection_results[0].boxes.data is not None:
        # Get the bounding boxes, track IDs, class indices, and confidence scores of detected objects
        bounding_boxes = detection_results[0].boxes.xyxy.cpu()
        tracked_ids = detection_results[0].boxes.id.int().cpu().tolist()
        class_indices = detection_results[0].boxes.cls.int().cpu().tolist()
        confidence_scores = detection_results[0].boxes.conf.cpu()

        # Draw the crossing line on the frame
        cv2.line(frame, (690, crossing_line_y), (1130, crossing_line_y), (0, 0, 255), 3)

        # Loop through each detected object
        for bounding_box, track_id, class_index, confidence in zip(bounding_boxes, tracked_ids, class_indices, confidence_scores):
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, bounding_box)
            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get the class name for the detected object
            class_name = class_names[class_index]
            
            # Draw a circle at the center of the detected object
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            # Annotate the detected object with its ID and class name
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if the object has crossed the detection line
            if center_y > crossing_line_y and track_id not in crossed_object_ids:
                # Mark the object ID as crossed and increment the count for its class
                crossed_object_ids.add(track_id)
                object_counts[class_name] += 1

        # Display the counts of detected objects on the frame
        y_offset = 30
        for class_name, count in object_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_offset += 30
    
    # Show the processed frame in a window
    cv2.imshow('Tracking and Counting', frame)
    # Exit the loop if the Escape key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
