#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copyreg import pickle
import csv
import copy
import argparse
import itertools
import pickle as pkl
from collections import Counter, deque
from charset_normalizer import detect

import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import hand_gesture_mediapipe
from utils import CvFpsCalc
from utils import HAND_LANDMARK
from model import KeyPointClassifier
from model import PointHistoryClassifier
import math
from numpy import angle

# DATA_FILENAME = "model/point_history_classifier/datasets/prediction_results.pkl" # The filename to store the prediction results and key points (if COLLECT_DATA=True)
# DATA_FILENAME = "finger_tip.pkl" # The filename to store the prediction results and key points (if COLLECT_DATA=True)
video_filename = "trimed_GH010026"
DATA_FILENAME = f"{video_filename}_fingertip.pkl" # The filename to store the prediction results and key points (if COLLECT_DATA=True)
DEFAULT_VIDEO = rf"G:\My Drive\CMU\Research\Summer_2022\Videos\trimed_videos\{video_filename}.MP4"
VISUALIZE_PROGRESS = True # whether to show the visualization during feature extraction process

PREDICTION_LABELS = ["CW", "CCW", "Check", "Cross", "Right", "Left", "None"]


def get_args():
    """
    Process input arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, help='path to the video file', default=DEFAULT_VIDEO)
    parser.add_argument("--width", type=int, help='cap width', default=960)
    parser.add_argument("--height", type=int, help='cap height', default=540)
    parser.add_argument("--store_file", default=DATA_FILENAME, type=str, 
                        help="output file containing stored index fingertip coordinates")
    parser.add_argument("--novisual", dest="visualProgress", default=VISUALIZE_PROGRESS, action='store_false')

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.75)

    args = parser.parse_args()

    return args

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

def draw_fps(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image

def main():

    # Argument parsing #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # load video ###############################################################
    print("trying to read video", args.video)
    cap = cv.VideoCapture(args.video)

    # Model load #############################################################
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # a list used to store the [x, y] position of the index finger in each frame
    indexPosList = []


    # load frames from the video and process one by one
    frameId = 0
    while True:
        fps = cvFpsCalc.get()
        print(f"frame {frameId} fps={fps}")

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()

        if not ret:
            # we don't need to visualize during finger tip extraction
            break

        image = rescale_frame(image, percent=50)

        if args.visualProgress:
            # image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True


        if results.multi_hand_landmarks is None:
            # there is no result in this frame
            indexPosList.append([0, 0])
            continue
        
        # the landmarks on each hand in the current frame
        hand_landmarks = results.multi_hand_landmarks[0]

        

        # get the index finger tip
        indexFingerTipPos = hand_landmarks.landmark[HAND_LANDMARK.INDEX_FINGER_TIP]
        indexPosList.append([indexFingerTipPos.x * args.width, indexFingerTipPos.y * args.height])

        print("pos", [indexFingerTipPos.x, indexFingerTipPos.y])

        if args.visualProgress:
            # display each landmark on the current frame
            mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            debug_image = draw_fps(debug_image, fps)

            # Screen reflection #############################################################
            cv.imshow('Index Finger Tip Collection', debug_image)
        
        
        frameId += 1

    # store the prediction result
    if args.store_file:
        data = {}
        data["prediction_label"] = PREDICTION_LABELS
        data["keypoint_pos"] = indexPosList
        data["prediction_rsts"] = [0] * len(indexPosList)

        print(f"prediction results of {len(indexPosList)} samples to:", args.store_file)
        with open(args.store_file, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()