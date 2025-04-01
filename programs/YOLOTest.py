from ultralytics import YOLO
import cv2

try:
    # Load model
    model = YOLO('../models/yolov8n.pt')
    
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open webcam")
        
    for result in model(source=0, stream=True, show=True):
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    cv2.destroyAllWindows()
