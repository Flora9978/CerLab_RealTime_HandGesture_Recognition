#%% environments
import os
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from collections import deque

#%% functions
def load_data(filename):
    with open(filename, "rb") as f:
        data = pkl.load(f)

    for key in data:
        print(key, ": ", len(data[key]))
    return data



#%% some constants
video_filename = "trimed_GH010026"
prediction_result_filename = f"{video_filename}_fingertip.pkl"
true_label_rst_file = f"{video_filename}_fingertip_true_label_test.txt"
NUM_OF_FRAMES_PER_PRED = 45
FORCE_RESTART = True # ignore previous true label

#%% loading data
data = load_data(prediction_result_filename)
print(data.keys())

#%% parsing data
prediction_labels = data["prediction_label"]
n_labels = len(prediction_labels)
n_samples = len(data["keypoint_pos"])
time_stamp = np.arange(0, n_samples) / NUM_OF_FRAMES_PER_PRED
keypoint_x = np.asarray([x for x, _ in data["keypoint_pos"]], dtype=int)
keypoint_y = np.asarray([y for _, y in data["keypoint_pos"]], dtype=int)
prediction = np.asarray(data["prediction_rsts"], dtype=int)
prediction_label = [prediction_labels[i] for i in prediction]

print("prediction_label: ", prediction_labels)
print("keypoint_pos x: ", keypoint_x)
print("keypoint_pos y: ", keypoint_y)
print("prediction_rsts: ", prediction)

banner = ",".join(str(i)+":" + label for i, label in enumerate(prediction_labels))

print("x", min(keypoint_x), max(keypoint_x))
print("y", min(keypoint_y), max(keypoint_y))

#%% getting min-max of the keypoint positions
x_min, x_max = np.min(keypoint_x), np.max(keypoint_x)
y_min, y_max = np.min(keypoint_y), np.max(keypoint_y)

#%% normalize
keypoint_x = keypoint_x - x_min
keypoint_y = keypoint_y - y_min

x_max -= x_min
y_max -= y_min

keypoint_x += x_max >> 1
keypoint_y += y_max >> 1

#%% display past 30 points
base = ord('0')
terminate_flag = False

# load pre-stored file
if not FORCE_RESTART and os.path.exists(true_label_rst_file):
    pre_stored_data = np.loadtxt(true_label_rst_file, dtype=int)
    start_id = pre_stored_data[0]
    true_label = pre_stored_data[1:].tolist()
else:
    true_label = [n_labels-1] * (NUM_OF_FRAMES_PER_PRED-1)
    start_id = NUM_OF_FRAMES_PER_PRED-1

frame_id = start_id
while frame_id < n_samples:

    x = keypoint_x[frame_id-NUM_OF_FRAMES_PER_PRED+1:frame_id+1]
    y = keypoint_y[frame_id-NUM_OF_FRAMES_PER_PRED+1:frame_id+1]
    #print(idx, label_names[y_dataset[idx]])
    progress_str = "frame {}/{}".format(frame_id, n_samples)

    img = np.ones((y_max<<1, x_max<<1, 3))*255
    for index in range(NUM_OF_FRAMES_PER_PRED):
        # cv2.circle(img, (x[index], y[index]), 1 + int(index / 2), (152, 251, 152), 2)
        cv2.circle(img, (x[index], y[index]), 1 + int(index / 2), (0,0,0), 2)
    
    cv2.putText(img, banner, (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, prediction_label[frame_id], (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, progress_str, (0, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # plt.close()
    cv2.imshow('Index Finger Tip Collection', img)

    while True:
        # if keyboard.read_key() == "a":
        k = cv2.waitKey(3)
        if k == 27:  # ESC
            terminate_flag = True
            break
        
        if base <= k < base + n_labels: # valid label in range [0, n_labels-1]
            our_label = k - base
            print("mark current label as ", prediction_labels[our_label])
            true_label.append(our_label)

            # save file in case
            np.savetxt(true_label_rst_file, [frame_id]+true_label, fmt='%d')

            frame_id += 1
            break

        if k == ord('C') or k == ord('c'):
            if frame_id > NUM_OF_FRAMES_PER_PRED:
                frame_id -= 1
                true_label.pop()

                # save file in case
                np.savetxt(true_label_rst_file, [frame_id]+true_label, fmt='%d')
                print("backtrack by one")
                break
    
    if terminate_flag:
        break

np.savetxt(true_label_rst_file, [frame_id]+true_label, fmt='%d')

# add the data to the pkl file
data["true_label"] = true_label
with open(prediction_result_filename, 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
    
# %%