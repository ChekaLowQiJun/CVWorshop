from ultralytics import YOLO
import cv2

model = YOLO("//Users/cheka/Documents/Projects/Workshop/models/bestHandMovement.pt") #Replace with the absolute path

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Perform prediction
    
    results = model(frame, conf = 0.68)

    # Draw predictions on the frame

    annotated_frame = results[0].plot()

    # Coordinate of Targets

    one_top_left = (500, 500)
    one_bottom_right = (700, 700)
    two_top_left = (702, 500)
    two_bottom_right = (902, 700)
    three_top_left = (904, 500)
    three_bottom_right = (1104, 700)
    four_top_left = (1106, 500)
    four_bottom_right = (1306, 700)

    # Define color (BGR format) and thickness

    green = (0, 255, 0)  
    red = (50, 50, 255)
    black = (0, 0, 0)
    pink = (203, 192, 255)
    blue = (255, 0, 0)
    thickness = 2

    # Draw the rectangle on the current frame

    cv2.rectangle(annotated_frame, one_top_left, one_bottom_right, green, thickness)
    cv2.rectangle(annotated_frame, two_top_left, two_bottom_right, red, thickness)
    cv2.rectangle(annotated_frame, three_top_left, three_bottom_right, black, thickness)
    cv2.rectangle(annotated_frame, four_top_left, four_bottom_right, pink, thickness)

    for obj in results[0].boxes.data:
        x1, y1, x2, y2 = obj[:4]  # Coordinates of the bounding box
    
        # Calculate the center of the bounding box

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
    
        # Define the size of the small rectangle to draw 

        small_rect_width = 20
        small_rect_height = 20
    
        # Calculate the top-left corner of the small rectangle

        small_rect_x1 = int(center_x - small_rect_width / 2)
        small_rect_y1 = int(center_y - small_rect_height / 2)   
    
        # Calculate the bottom-right corner of the small rectangle

        small_rect_x2 = int(center_x + small_rect_width / 2)
        small_rect_y2 = int(center_y + small_rect_height / 2)
    
        # Draw the small rectangle 

        # Add color labels in bottom right corner
        text_x = annotated_frame.shape[1] - 200  # Right aligned
        text_y = annotated_frame.shape[0] - 50   # Bottom aligned
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2

        cv2.rectangle(annotated_frame, (small_rect_x1, small_rect_y1), (small_rect_x2, small_rect_y2), (0, 0, 255), 2)

        if center_x < one_top_left[0] + 200 and center_x > one_top_left[0] and center_y < one_bottom_right[1] and center_y > one_bottom_right[1] - 200:
            cv2.putText(annotated_frame, "Green", (text_x, text_y), font, font_scale, green, font_thickness)
            
        elif center_x < two_top_left[0] + 200 and center_x > two_top_left[0] and center_y < two_bottom_right[1] and center_y > two_bottom_right[1] - 200:
            cv2.putText(annotated_frame, "Red", (text_x, text_y - 40), font, font_scale, red, font_thickness)
            
        elif center_x < three_top_left[0] + 200 and center_x > three_top_left[0] and center_y < three_bottom_right[1] and center_y > three_bottom_right[1] - 200:
            cv2.putText(annotated_frame, "Black", (text_x, text_y - 80), font, font_scale, black, font_thickness) 
            
        elif center_x < four_top_left[0] + 200 and center_x > four_top_left[0] and center_y < four_bottom_right[1] and center_y > four_bottom_right[1] - 200:
            cv2.putText(annotated_frame, "Pink", (text_x, text_y - 120), font, font_scale, pink, font_thickness)
            

    # Display the frame
    cv2.imshow("Colours!", annotated_frame)

    # Break loop on 'q' key press

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources

cap.release()
cv2.destroyAllWindows()
