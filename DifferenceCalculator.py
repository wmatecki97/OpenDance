from cv2 import sqrt
import numpy as np
import math

class DifferenceCalculator:
    @staticmethod
    def convert_to_polar(tensor, hip_idx=11, shoulder_idx=5):
        """
        Convert tensor coordinates to polar coordinates relative to the left hip,
        with distances normalized by the distance between left hip and left shoulder.

        Args:
            tensor (numpy.ndarray): Input tensor with (x, y, probability) values.
            hip_idx (int): Index of the left hip keypoint in the tensor.
            shoulder_idx (int): Index of the left shoulder keypoint in the tensor.

        Returns:
            numpy.ndarray: Tensor with (angle, normalized_distance, probability) polar coordinates.
        """
        hip_x, hip_y, _ = tensor[hip_idx]
        shoulder_x, shoulder_y, _ = tensor[ shoulder_idx]
        shoulder_to_hip_dist = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y)

        tensor_polar = np.zeros_like(tensor)

        for i in range(tensor.shape[0]):
            x, y, prob = tensor[i]
            dx = x - hip_x
            dy = y - hip_y
            if dx == 0:
                tensor_polar[i] = 0, np.hypot(dx, dy) / shoulder_to_hip_dist, prob
            else:
                tensor_polar[i] = (y - hip_y) / (x - hip_x) /2+0.5 , np.hypot(dx, dy) / shoulder_to_hip_dist, prob

        return tensor_polar

    @staticmethod
    def total_difference(tensor1, tensor2):
        # Convert inputs to numpy arrays if they are not already
        tensor1 = np.array(tensor1)
        tensor2 = np.array(tensor2)

        # Check if the tensors have the same shape
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensors must be of the same shape")

        # Convert tensors to polar coordinates
        tensor1_polar = DifferenceCalculator.convert_to_polar(tensor1)
        tensor2_polar = DifferenceCalculator.convert_to_polar(tensor2)

        angle_diff = 0
        dist_diff = 0
        total_diff = 0
        # Calculate the total difference
        valid_points = 0
        for i in range(len(tensor1_polar)):
            if tensor1_polar[i][2] > 0.1 and tensor2_polar[i][2]> 0.3 and i != 1 and  i!=  2 and i != 3 and i != 4:
                #find the coordinates of the 2 points on the lines with the formula y=a*x+b and calculate teh distance between them
                #b we set to 0
                #a is tensor[i][0]
                #distance is tensor[i][1]

                def points_at_distance(a, z):
                    # Calculate the x and y coordinates
                    x = z / math.sqrt(1 + a**2)
                    y = a * x
                    # Points (x, y) and (-x, -y) are z distance from the origin
                    return (abs(x), abs(y))
    
                a1 = tensor1_polar[i][0]
                a2 = tensor2_polar[i][0]
                distance1 = tensor1_polar[i][1]
                distance2 = tensor2_polar[i][1]
                x1, y1 = points_at_distance(a1, distance1)
                x2, y2 = points_at_distance(a2, distance2)

                distance_between_points = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                total_diff+=distance_between_points#maximum penalty is 1
                valid_points+=1
        average_diff=total_diff/valid_points
        max_diff = 2 * len(tensor1_polar)
        score =  (max_diff-total_diff)/max_diff-0.8
        return score