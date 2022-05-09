#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copyreg import pickle
import csv
import copy
import argparse
import itertools
import pickle as pkl
from collections import Counter
from collections import deque
from charset_normalizer import detect

import cv2 as cv
import numpy as np
import mediapipe as mp
import hand_gesture_mediapipe
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# NUM_OF_FRAMES_PER_PRED = 30 # number of frames to make a gesture prediction
NUM_OF_FRAMES_PER_PRED = 45 # number of frames to make a gesture prediction
TEST_MODE = True # True: test model, False: collect data only
COLLECT_DATA = True # True: store the prediction results, False: don't store (real application)
DATA_FILENAME = "prediction_results.pkl" # The filename to store the prediction results (if COLLECT_DATA=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default='')
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--store_result', default=COLLECT_DATA, action='store_true',
                        help="store prediction results")
    parser.add_argument("--store_file", default=DATA_FILENAME, type=str, help="output file containing stored prediction results")

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


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    import_video = False
    # Camera preparation ###############################################################
    if args.video != '':
        cap = cv.VideoCapture(args.video)
        import_video = True
    else:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

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

    if TEST_MODE:
        """Single Model"""
        point_history_classifier = PointHistoryClassifier("model/point_history_classifier/point_history_classifier_LSTM_ConquerCross.tflite")

        """Two models"""
        # point_history_classifier = PointHistoryClassifier("model/point_history_classifier/point_history_classifier_moreCross.tflite")
        # point_history_classifier_three_labels = PointHistoryClassifier("model/point_history_classifier/point_history_classifier_simple3class.tflite")
        three_labels_map_full_labels = [2, 3, 6] # 0: check mark (2), 1: cross (3), 2: none (6)

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [
            row[0] for row in csv.reader(f)
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            # 'model/point_history_classifier/point_history_classifier_label_simple3.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = [
            row[0] for row in csv.reader(f)
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # Coordinate history #################################################################
    history_length = NUM_OF_FRAMES_PER_PRED
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0


    # variable to store the processing history
    prediction_rst_dict = {
        "keypoint_pos": [],
        "prediction_label": point_history_classifier_labels,
        "prediction_rsts": [],
        "prediction_rsts_three_labels": [],
        "prediction_rsts_final": []
    }

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        # cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
        if not ret:
            continue
        if import_video:
            image = rescale_frame(image, percent=50)

        # image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # add static angle algo here
                debug_image, static_gesture = hand_gesture_mediapipe.detectWithDynamic(debug_image, hand_landmarks, mp_hands)
                if static_gesture == 'point':
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        if point_history:
                            point_history.append(point_history[-1])
                        else:
                            point_history.append([0, 0])
                    
                    # store the keypoint_position for performance analysis
                    prediction_rst_dict["keypoint_pos"].append(point_history[-1])

                    # Finger gesture classification
                    # finger_gesture_id = 0
                    finger_gesture_id = len(point_history_classifier_labels)-1
                    finger_gesture_id_three_label = 2

                    point_history_len = len(pre_processed_point_history_list)
                    if TEST_MODE and point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                        # finger_gesture_id_three_label_raw = point_history_classifier_three_labels(
                        #     pre_processed_point_history_list)
                        # finger_gesture_id_three_label = three_labels_map_full_labels[finger_gesture_id_three_label_raw]
                        

                    prediction_rst_dict["prediction_rsts"].append(finger_gesture_id)
                    prediction_rst_dict["prediction_rsts_three_labels"].append(finger_gesture_id)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    
                    finger_gesture_id_counter = Counter(finger_gesture_history)
                    most_common_fg_id = finger_gesture_id_counter.most_common()

                    # prediction from most_common
                    finger_id_prediction_final = most_common_fg_id[0][0]
                    # if finger_gesture_id_three_label in finger_gesture_id_counter:
                        # finger_id_prediction_final = finger_gesture_id_three_label
                    
                    prediction_rst_dict["prediction_rsts_final"].append(finger_id_prediction_final)
                    
                    # print("most commonly seen gestures (id, #)", [(point_history_classifier_labels[i], cnt) for i, cnt in most_common_fg_id])
                    
                    # Drawing part 
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    # only display the most commonly seen gesture in the last 
                    # NUM_OF_FRAMES_PER_PRED frames
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[finger_id_prediction_final],
                    )

                    debug_image = draw_point_history(debug_image, point_history)
                    # debug_image = draw_info(debug_image, fps, mode, number)
        else:
            point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    # store the prediction result
    if args.store_result:
        print("prediction results are stored to:", args.store_file)
        with open(args.store_file, 'wb') as f:
            pkl.dump(prediction_rst_dict, f, pkl.HIGHEST_PROTOCOL)

    cap.release()
    cv.destroyAllWindows()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history, teleportation_threshold=0.2):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    n = len(temp_point_history)

    for i in range(0, n-2, 2):
        diff_x = temp_point_history[i] - temp_point_history[i+2]
        diff_y = temp_point_history[i+1] - temp_point_history[i+3]
        dist = diff_x * diff_x + diff_y * diff_y
        if dist > (teleportation_threshold * teleportation_threshold):
            temp_point_history[i+2] = temp_point_history[i]
            temp_point_history[i+3] = temp_point_history[i+1]

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        print("save new samples to csv (label type {number})".format(number=number))
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):

    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        """
        index:

        """

        if index in {4, 8, 12, 16, 20}:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        else:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

        # if index == 0:  # 手首1
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 1:  # 手首2
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 2:  # 親指：付け根
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 3:  # 親指：第1関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 4:  # 親指：指先
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        # if index == 5:  # 人差指：付け根
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 6:  # 人差指：第2関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 7:  # 人差指：第1関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 8:  # 人差指：指先
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        # if index == 9:  # 中指：付け根
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 10:  # 中指：第2関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 11:  # 中指：第1関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 12:  # 中指：指先
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        # if index == 13:  # 薬指：付け根
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 14:  # 薬指：第2関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 15:  # 薬指：第1関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 16:  # 薬指：指先
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        # if index == 17:  # 小指：付け根
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 18:  # 小指：第2関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 19:  # 小指：第1関節
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        # if index == 20:  # 小指：指先
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
        #               -1)
        #     cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
