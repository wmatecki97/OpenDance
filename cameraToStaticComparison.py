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

def compare_frames(frame_video):
    global scores  # Declare scores as global to modify it within the function
    global outputPersons

    for i in range(camera_buffer_size):
        ret_cam, frame_cam = cap.read()
    ret_cam, frame_cam = cap.read()
    frame_cam = cv2.flip(frame_cam, 1)

    # Detect and plot the results for the camera frame
    outputPersons = detector.detect_and_plot(frame_cam, plot=False)

    # Detect and plot the results for the video frame
    outputVideo = detector.detect_and_plot(frame_video, plot=True)

    if len(outputPersons) > 0 and len(outputVideo) > 0:
        for i in range(min(len(outputPersons), len(scores))):
            diff_value = DifferenceCalculator.total_difference(outputPersons[i].keypoints, outputVideo[0].keypoints)
            scores[i] += diff_value*10
        if len(outputPersons) > len(scores):
            for i in range(len(scores), len(outputPersons)):
                diff_value = DifferenceCalculator.total_difference(outputPersons[i].keypoints, outputVideo[0].keypoints)
                scores.append(diff_value*10)
        elif len(outputPersons) < len(scores):
            scores = scores[:len(outputPersons)]
        print(f"Scores: {scores}")

video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
frame_counter = 0
start_time = time.time()
player = MediaPlayer(video_path)

# Create a named window and set it to fullscreen
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret_video, frame_video = video.read()

    # Display the scores on the video frame
    for i in range(min(len(scores), 6)):
        text = f"Score {i + 1}: {int(scores[i])}"
        cv2.putText(frame_video, text, (100, 300 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video frame
    cv2.imshow('Video', frame_video)
    if frame_counter % 15 == 0:
        if not ret_video:
            break
        thread = threading.Thread(target=compare_frames, args=(frame_video,))
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
cap.release()
video.release()
cv2.destroyAllWindows()
