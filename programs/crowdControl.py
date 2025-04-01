from ultralytics import YOLO
import cv2
import numpy as np
import os

def process_frame(frame, model):
    """Process a frame to detect humans and count them in quadrants"""
    height, width = frame.shape[:2]
    
    # Divide frame into 4 quadrants
    quadrants = [
        frame[0:height//2, 0:width//2],  # Top-left
        frame[0:height//2, width//2:width],  # Top-right
        frame[height//2:height, 0:width//2],  # Bottom-left
        frame[height//2:height, width//2:width]  # Bottom-right
    ]
    
    # Process each quadrant
    counts = [0, 0, 0, 0]
    for i, quadrant in enumerate(quadrants):
        results = model(quadrant)
        
        # Count humans (class 0 in COCO dataset)
        for box in results[0].boxes:
            if box.cls == 0:  # Human class
                counts[i] += 1
                
                # Draw bounding box (adjusted for quadrant position)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if i == 1 or i == 3:  # Right quadrants
                    x1 += width//2
                    x2 += width//2
                if i == 2 or i == 3:  # Bottom quadrants
                    y1 += height//2
                    y2 += height//2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "human", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw dividing lines between quadrants
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 2)
    cv2.line(frame, (0, height//2), (width, height//2), (255, 0, 0), 2)
    
    # Display counts for each quadrant with improved visibility
    cv2.putText(frame, f"Top-Left: {counts[0]}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Top-Right: {counts[1]}", (width//2 + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Bottom-Left: {counts[2]}", (10, height//2 + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Bottom-Right: {counts[3]}", (width//2 + 10, height//2 + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def main():
    model = YOLO('../models/yolov8n.pt')  # Load YOLOv8n model
    video_path = 'RawDataset/videoplayback.mp4'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    # Check if video is readable
    if not os.access(video_path, os.R_OK):
        print(f"Error: No read permissions for video file {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}. Possible reasons:")
        print("- File is corrupted")
        print("- Codec not supported")
        print("- Invalid video format")
        return
        
    # Print video properties for debugging
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame, model)
        cv2.imshow('Crowd Control', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
