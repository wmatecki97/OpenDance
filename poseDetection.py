import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
import numpy as np

def detect_and_plot(image_path):
    """
    Detect objects in an image and plot the results.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """

    model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
    movenet = model.signatures['serving_default']


    # Load the input image
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image, channels=3)
    image = tf.expand_dims(image, axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    # Download the model from TF Hub


    # Run model inference
    outputs = movenet(image)
    keypoints = outputs['output_0']

    # Process the output and plot the results
    image_np = image.numpy().squeeze()  # Remove the batch dimension
    image_np = np.asarray(image_np, dtype=np.uint8)  # Convert to uint8 for plotting

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the image
    ax.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # Plot the bounding boxes and keypoints
    for idx in range(keypoints.shape[1]):
        keypoint = keypoints[0, idx]
        if keypoint[-1] > 0.5:  # Filter out low confidence detections
            # Plot the bounding box
            x1, y1, x2, y2 = (keypoint[51] * 256, keypoint[50] * 256,
                               keypoint[53] * 256, keypoint[52] * 256)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, edgecolor='r'))

            # Plot the keypoints
            for i in range(0, 51, 3):
                x, y, score = keypoint[i + 1], keypoint[i], keypoint[i + 2]
                if score > 0.5:
                    ax.scatter(x * 256, y * 256, s=10, c='g')

    plt.show()
    

detect_and_plot('squat.jpeg')

cv2.waitKey(0)