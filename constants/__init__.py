import cv2
# Constants for configuration
VIDEO_FILE_PATH = "test/sample.mp4"  # Path to the video file
CROSSING_LINE_Y = 430                  # Y-coordinate of the crossing line
OBJECT_CLASSES = [1, 2, 3, 5, 6, 7]    # List of object class indices to track
LINE_COLOR = (0, 0, 255)                # Color of the crossing line (BGR)
CIRCLE_COLOR = (0, 0, 255)              # Color of the center circle (BGR)
RECTANGLE_COLOR = (0, 255, 0)           # Color of the bounding rectangle (BGR)
TEXT_COLOR = (255, 0, 255)              # Color of the text (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX         # Font type for text annotations
FONT_SCALE = 0.6                        # Font scale for text annotations
FONT_THICKNESS = 2                      # Font thickness for text annotations