# Create an instance of the MoveNetDetector class
import cv2
from poseDetection import MoveNetDetector


detector = MoveNetDetector(verbose=True)

img = cv2.imread('yoga_2.jpg')


output_frame = detector.detect_and_plot(img, plot=True)

cv2.imshow("Output", output_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()