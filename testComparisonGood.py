import cv2
from DifferenceCalculator import DifferenceCalculator
from poseDetection import MoveNetDetector

detector = MoveNetDetector(verbose=False)

# frame_cam = cv2.imread('compare/291.jpg')
# frame_video = cv2.imread('compare/292_vid.jpg')

# outputPerson = detector.detect_and_plot(frame_cam, plot=False)

# # Detect and plot the results for the video frame
# outputVideo = detector.detect_and_plot(frame_video, plot=True)

# cv2.waitKey(0)
# if len(outputPerson) > 0 and len(outputVideo) > 0:
#     diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
#     print(f"Difference: {diff_value}")


frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/352_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/351.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/382_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/381.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/412_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/411.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/442_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/441.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/352_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/351.jpg')
outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/352_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/351.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")

frame_cam = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/352_vid.jpg')
frame_video = cv2.imread('C:/Users/Wiktor/OneDrive/Pulpit/temp/diff_calc_data/great/351.jpg')

outputPerson = detector.detect_and_plot(frame_cam, plot=True)

# Detect and plot the results for the video frame
outputVideo = detector.detect_and_plot(frame_video, plot=False)

if len(outputPerson) > 0 and len(outputVideo) > 0:
    diff_value = DifferenceCalculator.total_difference(outputPerson[0].keypoints, outputVideo[0].keypoints)
    print(f"Difference: {diff_value}")