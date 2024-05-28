import cv2
import threading
from poseDetection import MoveNetDetector
from DifferenceCalculator import DifferenceCalculator

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Open the video
video_path = 'demo.mp4'
video = cv2.VideoCapture(video_path)

detector = MoveNetDetector(verbose=False)

frame_counter = 0
diff_value = 0

def compare_frames(frame_cam, frame_video):
    global diff_value  # Declare diff_value as global to modify it within the function

    # Detect and plot the results for the camera frame
    outputPerson = detector.detect_and_plot(frame_cam, plot=True)

    # Detect and plot the results for the video frame
    outputVideo = detector.detect_and_plot(frame_video, plot=False)

    if len(outputPerson) > 0 and len(outputVideo) > 0:
        diff_value = DifferenceCalculator.total_difference_first_two(outputPerson[0].keypoints, outputVideo[0].keypoints)
        print(f"Difference: {diff_value}")

while True:
    # Read a frame from the camera
    ret_cam, frame_cam = cap.read()

    # Read a frame from the video
    ret_video, frame_video = video.read()

    if not ret_cam or not ret_video:
        break

    # Display the difference value on the video frame
    text = f"Difference: {diff_value}"
    cv2.putText(frame_video, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video frame
    cv2.imshow('Video', frame_video)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

    # Perform comparison every 15 frames
    if frame_counter % 15 == 0:
        # Schedule the comparison on an independent thread
        thread = threading.Thread(target=compare_frames, args=(frame_cam, frame_video))
        thread.start()

# Release the camera, video, and close all windows
cap.release()
video.release()
cv2.destroyAllWindows()