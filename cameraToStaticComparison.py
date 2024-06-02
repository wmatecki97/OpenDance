import os
import time
from typing import List
import cv2
import threading

import numpy as np
from poseDetection import MoveNetDetector, Person
from DifferenceCalculator import DifferenceCalculator
from ffpyplayer.player import MediaPlayer

offset = -0.3  # todo some slider or sth
camera_buffer_size = 5

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Open the video
video_path = 'demo2.mp4'
video = cv2.VideoCapture(video_path)

detector = MoveNetDetector(verbose=False)
outputPersons: List[Person] = []
frame_counter = 0
scores = []
detection_in_progress=False

def get_points_by_difference_score(score):
    print(score)
    if score < 0.1:
        return 2
    elif score < 0.2:
        return 1
    elif score < 0.3:
        return 0.5
    else:
        return 0

def compare_frames(frame_video):
    global scores  # Declare scores as global to modify it within the function
    global outputPersons
    global detection_in_progress

    for i in range(camera_buffer_size):
        ret_cam, frame_cam = cap.read()
    ret_cam, frame_cam = cap.read()
    frame_cam = cv2.flip(frame_cam, 1)
    
    # os.makedirs('compare', exist_ok=True)
    # cv2.imwrite(f'compare/{frame_counter}.jpg', frame_cam)
    # cv2.imwrite(f'compare/{frame_counter}_vid.jpg', frame_video)
    # Detect and plot the results for the camera frame
    outputPersons = detector.detect_and_plot(frame_cam, plot=False)
    # Detect and plot the results for the video frame
    outputVideo = detector.detect_and_plot(frame_video, plot=True)

    if len(outputPersons) > 0 and len(outputVideo) > 0:
        for i in range(min(len(outputPersons), len(scores))):
            diff_value = DifferenceCalculator.total_difference(outputPersons[i].keypoints, outputVideo[0].keypoints)
            scores[i] += get_points_by_difference_score(diff_value)
        if len(outputPersons) > len(scores):
            for i in range(len(scores), len(outputPersons)):
                diff_value = DifferenceCalculator.total_difference(outputPersons[i].keypoints, outputVideo[0].keypoints)
                scores.append(get_points_by_difference_score(diff_value))

        print(f"Scores: {scores}")
    detection_in_progress=False


def resize_and_pad(image, size):
    h, w = image.shape[:2]
    if h==0 or w ==0:
        return image
    target_h, target_w = size
    
    # Calculate scale and new size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate top-left coordinates to center the image
    top_left_y = (target_h - new_h) // 2
    top_left_x = (target_w - new_w) // 2
    
    # Place the resized image on the canvas
    canvas[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_image
    
    return canvas

# Initialize video capture
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
frame_counter = 0
start_time = time.time()
player = MediaPlayer(video_path)

# Initialize camera capture
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create a named window and set it to fullscreen
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Detection flag
detection_in_progress = False

while True:
    ret_video, frame_video = video.read()
    ret_camera, frame_camera = camera.read()
    frame_camera = cv2.flip(frame_camera, 1)
    clean_video_frame = frame_video.copy()
    if not ret_video or not ret_camera:
        break

    cam_h, cam_w, _ = frame_camera.shape

    for person in outputPersons:
        if person.probability < 0.3:
            continue

        rect = person.rectangle

        # Convert percentages to coordinates based on preprocessed image
        x1, y1, x2, y2 = rect

        # Add padding
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(cam_w, x2 + padding)
        y2 = min(cam_h, y2 + padding)

        # Crop the camera frame
        cropped_camera_frame = frame_camera[y1:y2, x1:x2]

        # Resize cropped frame to 160x120 with padding
        small_cropped_frame = resize_and_pad(cropped_camera_frame, (160, 120))
        
        # Define the position for the small cropped camera frame
        top_left_y = max(0,frame_video.shape[0] - small_cropped_frame.shape[0] - 10 ) # 10 pixels from bottom
        top_left_x = 10  # 10 pixels from left

        # Overlay the small cropped camera frame on the video frame
        frame_video[top_left_y:top_left_y + small_cropped_frame.shape[0], top_left_x:top_left_x + small_cropped_frame.shape[1]] = small_cropped_frame

        # Increment position for next person's cropped image to avoid overlap
        top_left_y -= (small_cropped_frame.shape[0] + 10)
        if top_left_y < 0:
            break

    # Display the scores on the video frame
    for i in range(min(len(scores), 6)):
        text = f"Score {i + 1}: {scores[i]}"
        cv2.putText(frame_video, text, (100, 300 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video frame
    cv2.imshow('Video', frame_video)

    if frame_counter % 15 == 0:
        if not detection_in_progress:
            detection_in_progress = True
            thread = threading.Thread(target=compare_frames, args=(clean_video_frame,))
            thread.start()

    frame_counter += 1
    curr_time = time.time()
    elapsed_time = curr_time - start_time

    displayed_frames_time = frame_counter / fps

    TIME_TO_SLEEP_TO_MATCH_AUDIO = max(0.001, displayed_frames_time - elapsed_time - offset)

    # Press 'q' to quit
    if cv2.waitKey(int((TIME_TO_SLEEP_TO_MATCH_AUDIO) * 1000)) & 0xFF == ord('q'):
        break

# Release the camera, video, and close all windows
camera.release()
video.release()
cv2.destroyAllWindows()