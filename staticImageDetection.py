# Create an instance of the MoveNetDetector class
import cv2
from poseDetection import MoveNetDetector


detector = MoveNetDetector(verbose=True)

img = cv2.imread('yoga_2.jpg')


output_frame = detector.detect_and_plot(img, plot=False)


cv2.waitKey(0)
cv2.destroyAllWindows()