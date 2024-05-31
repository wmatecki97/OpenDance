import os
import time
from typing import List, Tuple
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

class Person(BaseModel):
    keypoints: List[Tuple[int, int, int, int]] = []
    rectangle:Tuple[int, int, int, int] = (0, 0, 0, 0)
    probability: float = 0.0

verify = True
class MoveNetDetector:
    def __init__(self, model_path='model', verbose=True):
        self.verbose = verbose
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
        movenet = model.signatures['serving_default']
        return movenet

    def preprocess_image(self, image):
        image = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.uint8), axis=0)
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
        return image

    def detect_and_plot(self, image, plot=True):
        start_time = time.time()
        image = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        if self.verbose:
            print(f"Preprocess: {preprocess_time:.3f} s")

        start_time = time.time()
        outputs = self.model(image)
        model_inference_time = time.time() - start_time
        if self.verbose:
            print(f"Model Inference: {model_inference_time:.3f} s")

        keypoints = outputs['output_0']

        start_time = time.time()
        image_np = image.numpy().squeeze()
        image_np = np.asarray(image_np, dtype=np.uint8)
        image_height, image_width, _ = image_np.shape
        people = []
        for idx in range(keypoints.shape[1]):
            keypoint = keypoints[0, idx]
            if keypoint[-1] > 0.1:
                person = Person(probability=keypoint[-1])
                x1, y1, x2, y2 = (keypoint[52] * image_width, keypoint[51] * image_height,
                                keypoint[54] * image_width, keypoint[53] * image_height)
        
                if plot:
                    cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                person.rectangle = (x1, y1, x2, y2)
                for i in range(0, 51, 3):
                    x, y, score = keypoint[i + 1], keypoint[i], keypoint[i + 2]
                    person.keypoints.append((x,y,score))

                    if score > 0.1 and not verify or i==5*3 or i==11*3:
                        cv2.circle(image_np, (int(x * image_width), int(y * image_height)), 5, (0, 255, 0), -1)
                people.append(person)

        post_process_time = time.time() - start_time
        if self.verbose:
            print(f"Post-process: {post_process_time:.3f} s")
        if plot:
            cv2.imshow("Output", image_np)
        return people