# Create an instance of the MoveNetDetector class
import cv2
import numpy as np
from poseDetection import MoveNetDetector

cv2.waitKey(0)
cv2.destroyAllWindows()
detector = MoveNetDetector(verbose=False)

img = cv2.imread('yoga_2.jpg')
compareFrame = detector.detect_and_plot(img, plot=False)

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def total_difference_first_two(tensor1, tensor2):
    # Convert inputs to numpy arrays if they are not already
    tensor1 = np.array(tensor1)
    tensor2 = np.array(tensor2)

    # Check if the tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape")

    # Slice tensors to only include the first two elements of each tuple
    tensor1_sliced = tensor1[:, :2]
    tensor2_sliced = tensor2[:, :2]

    # Calculate the total difference
    total_diff = np.sum(np.abs(tensor1_sliced - tensor2_sliced))

    return total_diff

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and plot the results
    outputPerson = detector.detect_and_plot(frame, plot=True)
    
        # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if(len(outputPerson) == 0 or len(compareFrame)==0):
        continue
    diff = total_difference_first_two(outputPerson[0].keypoints, compareFrame[0].keypoints)
    print(diff)

    


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()