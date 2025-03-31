import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO pose estimation model
model = YOLO('models/yolo11n-pose.pt')

# Pose keypoints indices (YOLO format)
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9
RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10

def is_l_flex(pose):
    """Detect L-shaped bicep flex pose"""
    # Get relevant keypoints
    left_shoulder = pose[LEFT_SHOULDER]
    left_elbow = pose[LEFT_ELBOW]
    left_wrist = pose[LEFT_WRIST]
    
    right_shoulder = pose[RIGHT_SHOULDER]
    right_elbow = pose[RIGHT_ELBOW]
    right_wrist = pose[RIGHT_WRIST]
    
    # Check if either arm is in L shape
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Wider angle range for more flexibility
    return (80 < left_angle < 100) or (80 < right_angle < 100)

def draw_keypoints(frame, pose):
    """Draw pose keypoints on frame"""
    for i, point in enumerate(pose):
        if i in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 
                 RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    counter = 0
    prev_state = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run pose estimation
        results = model(frame)
        
        if results[0].keypoints is not None:
            pose = results[0].keypoints.xy[0].cpu().numpy()
            
            # Only process if we have valid keypoints
            if len(pose) > 0:
                # Check for L flex pose
                current_state = is_l_flex(pose)
                
                # Increment counter on state change
                if current_state and not prev_state:
                    counter += 1
                prev_state = current_state
            else:
                current_state = False
            
            # Draw keypoints and counter
            draw_keypoints(frame, pose)
            cv2.putText(frame, f"Count: {counter}", 
                        (frame.shape[1] - 200, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Display frame
        cv2.imshow('Bicep Flex Counter', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
