# Create an instance of the MoveNetDetector class
import cv2
from poseDetection import MoveNetDetector


detector = MoveNetDetector(verbose=True)

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and plot the results
    output_frame = detector.detect_and_plot(frame, plot=True)

    # Display the output frame
    cv2.imshow("Output", output_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()