'''
1st Step for data cleaning: manually find all the bad samples and create a text file to store them.
ATTENTION!!! In this elementary file, you can NOT labelling wrong! Or you need to do it all over again (*^_^*) cheers!
Also, a temp.png file will be save in the current directory after running this code.
'''

import csv

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

dataset = 'model/point_history_classifier/datasets/point_history.csv'
label_names = ["Clockwise", "Counter Clockwise", "Checkmark", "Cross", "Right", "Left"]

NUM_CLASSES = 6
TIME_STEPS = 30 #16 or 30
DIMENSION = 2
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# filter 
teleportation_threshold = 0.2 # allowed maximum teleportation distance
for i in range(TIME_STEPS-1):

    diff_x = X_dataset[:, 2*i] - X_dataset[:, 2*i+2]
    diff_y = X_dataset[:, 2*i+1] - X_dataset[:, 2*i+3]
    dist = diff_x * diff_x + diff_y * diff_y
    filt = dist > (teleportation_threshold * teleportation_threshold)
    X_dataset[filt, 2*i+2] = X_dataset[filt, 2*i]
    X_dataset[filt, 2*i+3] = X_dataset[filt, 2*i+1]

# remove records that have only outliers
amplitute = np.sum(np.abs(X_dataset), axis=1)
filt = amplitute > 0.01
X_dataset = X_dataset[filt]
y_dataset = y_dataset[filt]

import keyboard
import cv2
idx = np.random.randint(0, 2520)
badSampleList2 = []
startIdx = 0
for idx in range(startIdx, len(X_dataset)):
    print(label_names[y_dataset[idx]])
    x = X_dataset[idx,::2]
    y = X_dataset[idx,1::2]
    #print(idx, label_names[y_dataset[idx]])
    plt.clf()
    img = plt.plot(x, y)
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    # plt.savefig("temp.png")
    img = cv2.imread("temp.png")
    cv2.imshow(label_names[y_dataset[idx]], img)
    # plt.title("label="+label_names[y_dataset[idx]])

    # plt.show()
    # plt.close()
    while True:
        # if keyboard.read_key() == "a":
        k = cv2.waitKey(3)
        if k == ord('a'): # means accept
            print("record", idx, label_names[y_dataset[idx]], " is good!")
            break
        # if keyboard.read_key() == "d":
        if k == ord('d'): # means delete
            print("record", idx, label_names[y_dataset[idx]], " will be deleted!")
            badSampleList2.append(idx)
            np.savetxt("model/test.txt", badSampleList2)
            
            break
    print("we will delete the following data")
    for idx in badSampleList2:
        print(idx)
    