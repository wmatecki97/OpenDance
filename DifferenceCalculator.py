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
            tensor_polar[i] = np.arctan2(dy, dx), np.hypot(dx, dy) / shoulder_to_hip_dist, prob

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

        # Calculate the total difference
        angle_diff = np.sum(np.abs(tensor1_polar[ :, 0] - tensor2_polar[ :, 0]))
        dist_diff = np.sum(np.abs(tensor1_polar[ :, 1] - tensor2_polar[ :, 1]))
        total_diff = angle_diff + dist_diff

        return total_diff