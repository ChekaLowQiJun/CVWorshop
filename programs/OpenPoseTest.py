from ultralytics import YOLO
import cv2

try:
    # Load model
    model = YOLO('../models/yolo11n-pose.pt')
    
    # Start video capture
    for result in model(source=0, stream=True, show=True):
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    cv2.destroyAllWindows()
