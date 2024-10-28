import cv2
import numpy as np

# Set up the webcam
cap = cv2.VideoCapture(0)

# Laser pointer color range (adjust as needed)
lower_red = np.array([0, 0, 255])
upper_red = np.array([0, 0, 255])

while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Color thresholding
    mask = cv2.inRange(frame, lower_red, upper_red)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculate the centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Mark the laser point in the image
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            
            # Calculate distance (assuming known laser point size)
            # Adjust this part based on actual requirements
            distance = 200  # Example distance in cm
            
            # Display the distance
            cv2.putText(frame, f"Distance: {distance} cm", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Laser Pointer Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()