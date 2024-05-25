import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

def detect_and_plot():
    """
    Detect objects in the camera feed and display the results with FPS.
    """


    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera
    model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
    movenet = model.signatures['serving_default']
    prev_time = 0
    fps = 0

    while True:
        start_time = time.time()  # Start measuring time for reading the frame
        ret, frame = cap.read()
        read_frame_time = time.time() - start_time
        if not ret:
            break

        # Preprocess the input image
        start_time = time.time()  # Start measuring time for preprocessing
        image = tf.expand_dims(tf.convert_to_tensor(frame, dtype=tf.uint8), axis=0)
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
        preprocess_time = time.time() - start_time

        # Run model inference
        start_time = time.time()  # Start measuring time for model inference
        outputs = movenet(image)
        model_inference_time = time.time() - start_time

        keypoints = outputs['output_0']

        # Process the output and display the results
        start_time = time.time()  # Start measuring time for post-processing
        image_np = image.numpy().squeeze()  # Remove the batch dimension
        image_np = np.asarray(image_np, dtype=np.uint8)

        # Calculate and display FPS
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time
        fps = 1 / elapsed_time
        cv2.putText(image_np, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Plot the bounding boxes and keypoints
        for idx in range(keypoints.shape[1]):
            keypoint = keypoints[0, idx]
            if keypoint[-1] > 0.5:  # Filter out low confidence detections
                # Plot the bounding box
                x1, y1, x2, y2 = (keypoint[51] * 256, keypoint[50] * 256,
                                   keypoint[53] * 256, keypoint[52] * 256)
                cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Plot the keypoints
                for i in range(0, 51, 3):
                    x, y, score = keypoint[i + 1], keypoint[i], keypoint[i + 2]
                    if score > 0.5:
                        cv2.circle(image_np, (int(x * 256), int(y * 256)), 5, (0, 255, 0), -1)

        cv2.imshow("Output", image_np)

        post_process_time = time.time() - start_time
        total_time = read_frame_time + preprocess_time + model_inference_time + post_process_time
        print(f"Read frame: {read_frame_time:.3f} s, Preprocess: {preprocess_time:.3f} s, Model Inference: {model_inference_time:.3f} s, Post-process: {post_process_time:.3f} s, Total: {total_time:.3f} s")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

detect_and_plot()