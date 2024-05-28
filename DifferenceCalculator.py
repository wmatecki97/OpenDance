import numpy as np

class DifferenceCalculator:
    @staticmethod
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