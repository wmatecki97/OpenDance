import cv2
from DifferenceCalculator import DifferenceCalculator
from poseDetection import MoveNetDetector

frame_cam = cv2.imread('compare/291frame_video.jpg')
frame_video = cv2.imread('compare/290frame_cam.jpg')


detector = MoveNetDetector(verbose=False)

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

cv2.waitKey(0)
if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")