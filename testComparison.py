import cv2
from DifferenceCalculator import DifferenceCalculator
from poseDetection import MoveNetDetector

detector = MoveNetDetector(verbose=False)

frame_cam = cv2.imread('compare/1279.jpg')
frame_video = cv2.imread('compare/1279_vid.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

cv2.waitKey(0)
if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    diff_value2 = DifferenceCalculator.total_difference(outputPerson[1].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")


frame_cam = cv2.imread('compare/1279.jpg')
frame_video = cv2.imread('compare/1234_vid.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

cv2.waitKey(0)
if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")