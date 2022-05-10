import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from collections import Counter, defaultdict

def rotationMatrix(angle):
    """
    Return a 2d rotation matrix

    input:
        angle: float, rotation angle in degree
    
    return:
        rotationMatrix: 3x3 matrix including homogeneous coordinates
    """
    r = Rotation.from_quat([0, 0, np.sin(np.pi/(360/angle)), np.cos(np.pi/(360/angle))])
    return r.as_matrix()

def translateMatrix(x_dist, y_dist):
    """
    Return a 2d transition matrix

    input:
        x_dist: delta on x-axis
        y_dist: delta on y-axis
    
    return:
        transitionMatrix: 3x3 matrix including homogeneous coordinates
    """
    t = np.identity(3)
    t[0,2] = x_dist
    t[1,2] = y_dist
    return t

def scaleMatrix(x_scale, y_scale):
    """
    Return a 2d scaling matrix

    input:
        x_scale: scale factor on x-axis
        y_scale: scale factor on y-axis
    
    return:
        scaleMatrix: 3x3 matrix including homogeneous coordinates
    """
    s = np.identity(3)
    s[0, 0] = x_scale
    s[1, 1] = y_scale
    return s

def addGaussian(X_dataset, y_dataset, mu, sigma):
    """
    Add Gaussian noise to the X-dataset.

    input:
        X_dataset: 
        y_dataset:
        mu: float, mean of Gaussian noise
        sigma: standard derivation of Gaussian noise
    
    return:
        X_dataset 
        y_dataset
    """
    noise = np.random.normal(mu, sigma, X_dataset.shape)
    X_dataset += noise
    return X_dataset, y_dataset


def transformDataset(data, label, H_dict):
    TIME_STAMP = data.shape[1] // 2

    for sample_idx in range(len(data)):
        x = data[sample_idx,::2]
        y = data[sample_idx,1::2]
        l = label[sample_idx]


        one_sample = np.vstack([x, y, np.ones(TIME_STAMP,)]).T

        for H in H_dict[l]:
            # do transformation to this sample
            transed_sample = np.dot(one_sample, H)
            transed_sample[:, :2] /= transed_sample[:, 2:]
            transed_sample = np.reshape(transed_sample[:,:2], (1, TIME_STAMP*2))
            data = np.append(data, transed_sample, axis=0)
            label = np.append(label, label[sample_idx])
    return data, label

def generateTransitionMatrixH(
    CCW_n_CW_dup_times = 4,          # number of fake copies by duplicating original samples
    cross_dup_times = 6,             # number of fake copies by duplicating original samples
    check_dup_times = 4,             # number of fake copies by duplicating original samples
    left_n_right_dup_times = 4      # number of fake copies by duplicating original samples)
    ):
    """
    
    return:
        H_dict: {0:CCW_n_CW_dup_H, 1:CCW_n_CW_dup_H, 2:check_dup_H, 3:cross_dup_H, 4:left_n_right_dup_H, 5:left_n_right_dup_H}
    """

    H_rotation_list = [rotationMatrix(angle) for angle in range(60, 360, 60)]
    H_translation_list = [translateMatrix(dx, dy) for dx in np.arange(-0.2, 0.2, 0.05) for dy in np.arange(-0.2, 0.2, 0.05)]
    H_scale_list = [scaleMatrix(x_scale, y_scale) for x_scale in np.arange(0.8, 1.2, 0.05) for y_scale in np.arange(0.8, 1.2, 0.05)]

    CCW_n_CW_dup_H_random_idx = [
        np.random.permutation(len(H_rotation_list))[:CCW_n_CW_dup_times], 
        np.random.permutation(len(H_translation_list))[:CCW_n_CW_dup_times],
        np.random.permutation(len(H_scale_list))[:CCW_n_CW_dup_times]
    ]
    cross_dup_H_random_idx = [
        np.random.permutation(len(H_rotation_list))[:cross_dup_times],
        np.random.permutation(len(H_translation_list))[:cross_dup_times],
        np.random.permutation(len(H_scale_list))[:cross_dup_times]
    ]
    check_dup_H_random_idx = [
        np.random.permutation(len(H_translation_list))[:check_dup_times],
        np.random.permutation(len(H_scale_list))[:check_dup_times]
    ]
    left_n_right_random_idx = [
        np.random.permutation(len(H_translation_list))[:left_n_right_dup_times],
        np.random.permutation(len(H_scale_list))[:left_n_right_dup_times]
    ]

    CCW_n_CW_dup_H = []
    for rotation_H_idx, translation_H_idx, scale_H_idx in zip(CCW_n_CW_dup_H_random_idx[0], CCW_n_CW_dup_H_random_idx[1], CCW_n_CW_dup_H_random_idx[2]):
        CCW_n_CW_dup_H.append(H_rotation_list[rotation_H_idx].dot(H_translation_list[translation_H_idx]).dot(H_scale_list[scale_H_idx]))

    cross_dup_H = []
    for rotation_H_idx, translation_H_idx, scale_H_idx in zip(cross_dup_H_random_idx[0], cross_dup_H_random_idx[0], cross_dup_H_random_idx[0]):
        cross_dup_H.append(H_rotation_list[rotation_H_idx].dot(H_translation_list[translation_H_idx]).dot(H_scale_list[scale_H_idx]))

    check_dup_H = []
    for translation_H_idx, scale_H_idx in zip(check_dup_H_random_idx[0], check_dup_H_random_idx[1]):
        check_dup_H.append(H_translation_list[translation_H_idx].dot(H_scale_list[scale_H_idx]))

    left_n_right_dup_H = []
    for translation_H_idx, scale_H_idx in zip(left_n_right_random_idx[0], left_n_right_random_idx[1]):
        left_n_right_dup_H.append(H_translation_list[translation_H_idx].dot(H_scale_list[scale_H_idx]))


    H_dict = {0:CCW_n_CW_dup_H, 1:CCW_n_CW_dup_H, 2:check_dup_H, 3:cross_dup_H, 4:left_n_right_dup_H, 5:left_n_right_dup_H}
    return H_dict


def briefOfY(y_dataset, label_names=None):
    '''
    Ouput example:
    {'Clockwise': 3995, 'Counter Clockwise': 3790, 'Checkmark': 3540, 'Cross': 3846, 'Right': 2965, 'Left': 3420, 'None': 0}
    '''
    counter = Counter(y_dataset)
    label_count = {}
    if label_names:
        for key_id, key in enumerate(label_names):
            label_count[key] = counter[key_id]
    else:
        label_count = counter
    print(label_count)
    # remember, the Cross gesture may have been deleted, but we are too lazy to change the label name


def createRandomWalkingData(random_walking, time_steps, maxMobiliarbus=0.02):
    random_walk_data = []
    for _ in range(random_walking):
        samples = [0,0]
        for _ in range(time_steps-1):
            dx, dy = random.uniform(-1, 1) * maxMobiliarbus, random.uniform(-1, 1) * maxMobiliarbus
            newx, newy = samples[-2]+dx, samples[-1]+dy
            while not (-0.5<=newx<=0.5 and -0.5 <=newy<=0.5):
                dx, dy = random.uniform(-1, 1) * maxMobiliarbus, random.uniform(-1, 1) * maxMobiliarbus
                newx, newy = samples[-2]+dx, samples[-1]+dy
            samples += [newx, newy]
        random_walk_data.append(np.asarray(samples))
    random_walk_data = np.asarray(random_walk_data)
    return random_walk_data

def createSuddenData(sudden_points, time_steps):
    sudden_points_data = []
    for _ in range(sudden_points):
        sample = np.random.rand(time_steps*2)*2 - 0.55
        sudden_points_data.append(sample)
    sudden_points_data = np.asarray(sudden_points_data)
    return sudden_points_data

def createStopMoveData(stop_moving, time_steps, stopMobiliarbus=5e-3):
    stop_moving_data = []
    for sample_num in range(stop_moving):
        samples = [0,0]
        for _ in range(time_steps-1):
            dx, dy = random.uniform(-1, 1) * stopMobiliarbus, random.uniform(-1, 1) * stopMobiliarbus
            newx, newy = samples[-2]+dx, samples[-1]+dy
            while not (-0.5 <= newx <= 0.5 and -0.5 <= newy <= 0.5):
                dx, dy = random.uniform(-1, 1) * stopMobiliarbus, random.uniform(-1, 1) * stopMobiliarbus
                newx, newy = samples[-2]+dx, samples[-1]+dy
            samples += [newx, newy]
        stop_moving_data.append(np.asarray(samples))
    stop_moving_data = np.asarray(stop_moving_data)
    return stop_moving_data

def createNoneTypeData(TIME_STEPS, nonetype_label, random_walking, sudden_points, stop_moving):
    # print("sudden_points has been removed")

    random_walk_data = createRandomWalkingData(random_walking, TIME_STEPS)
    sudden_points_data = createSuddenData(sudden_points, TIME_STEPS)
    stop_moving_data = createStopMoveData(stop_moving, TIME_STEPS)
    nonetype_labels = nonetype_label * np.ones((random_walking+ sudden_points + stop_moving), dtype=int)

    X_dataset = np.vstack([random_walk_data, sudden_points_data, stop_moving_data])
    # X_dataset = np.vstack([random_walk_data, stop_moving_data])
    y_dataset = nonetype_labels
    return X_dataset, y_dataset

def appendTwoDatasets(X1, X2, y1, y2):
    X_dataset = np.vstack([X1, X2])
    y_dataset = np.hstack([y1, y2])
    return X_dataset, y_dataset

def display(data_set, dataset_label, label_dataset=None, label_names=None, idx=None):
    '''
    display a sample
    '''
    if idx is None:
        idx = np.random.randint(0, data_set.shape[0])
    print(dataset_label+"=", data_set.shape)
    if label_dataset is not None:
        if label_names is None:
            print("label", label_dataset[idx])
        else:
            print("label", label_names[label_dataset[idx]])
    x = data_set[idx,::2]
    y = data_set[idx,1::2]
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    print(idx, dataset_label)
    plt.plot(x, y)
    return idx

'''
========================= PADDING Samples Together!!! =========================
'''

def alignRandomSample(X_dataset, RandomDataset, isBefore=True):
    """
    Translate the RandomDataset such that its first (or last) coordinate is the same as the X_dataset's last (or first) coordinate.
    """
    if isBefore:
        # adjust the last point in RandomBefore to make sure it's the same as the 
        # first point in X_dataset
        RandomDataset[:, ::2] = (RandomDataset[:, ::2].T + X_dataset[:, 0]- RandomDataset[:, -2]).T
        RandomDataset[:, 1::2] = (RandomDataset[:, 1::2].T + X_dataset[:, 1]- RandomDataset[:, -1]).T
        
    else: # after
        # do the same for RandomAfter, but this time, make sure it's first point 
        # is at the same position as the last point in X_dataset
        RandomDataset[:, ::2] = (RandomDataset[:, ::2].T +X_dataset[:, -2] -RandomDataset[:, 0]).T
        RandomDataset[:, 1::2] = (RandomDataset[:, 1::2].T + X_dataset[:, -1] - RandomDataset[:, 1]).T
    
    return RandomDataset


def addRandomWalkBeforeAfter(X_dataset, nSamplesBefore, nSamplesAfter):
    nSamples = X_dataset.shape[0]
    
    RandomBefore = createRandomWalkingData(nSamples, nSamplesBefore+1, maxMobiliarbus=0.05)
    RandomAfter = createRandomWalkingData(nSamples, nSamplesAfter+1, maxMobiliarbus=0.05)

    RandomBefore = alignRandomSample(X_dataset, RandomBefore, isBefore=True)
    RandomAfter = alignRandomSample(X_dataset, RandomAfter, isBefore=False)
    
    # ignore the last point in RandomBefore and the first point in RandomAfter
    X_dataset = np.hstack([RandomBefore[:, :-2], X_dataset, RandomAfter[:, 2:]])

    return X_dataset

def addRandomSamplesBeforeAfter_v1(X_dataset, nSamplesBefore, nSamplesAfter):
    """
    For each sample, randomly select two additional samples from the dataset, and truncate out the last nSamplesBefore frames from the first random sample and truncate out the first nSamplesAfter frames from the second sample. Concatenate the three pieces and make the original sample in the middle.

    nSamplesBefore and nSamplesAfter are two fixed numbers.

    """
    nSamples = X_dataset.shape[0]
    
    randomSamples = np.random.randint(nSamples, size=nSamples)

    RandomBefore = X_dataset[randomSamples].copy()
    RandomAfter = X_dataset[randomSamples].copy()

    RandomBefore = RandomBefore[:, -nSamplesBefore*2-2:]
    RandomAfter = RandomAfter[:, :nSamplesAfter*2+2]

    RandomBefore = alignRandomSample(X_dataset, RandomBefore, isBefore=True)
    RandomAfter = alignRandomSample(X_dataset, RandomAfter, isBefore=False)
    
    # ignore the last point in RandomBefore and the first point in RandomAfter
    X_dataset = np.hstack([RandomBefore[:, :-2], X_dataset, RandomAfter[:, 2:]])

    return X_dataset

def addRandomSamplesBeforeAfter_v2(X_dataset, targetFrameLength, leftFrameLength, rightFrameLength):
    """
    For each sample, randomly select two additional samples from the dataset, and truncate out the last nSamplesBefore frames from the first random sample and truncate out the first nSamplesAfter frames from the second sample. Concatenate the three pieces and make the original sample in the middle.

    nSamplesBefore is a random number randomly sampled from [leftEdge, rightEdge] (inclusive) following uniform distribution.

    """
    nSamples = X_dataset.shape[0]
    TIME_STAMP = X_dataset.shape[1] // 2
    
    # randomly selected sample index of shape(nSamples,)
    randomSamplesBefore = np.random.randint(nSamples, size=nSamples)
    randomSamplesAfter = np.random.randint(nSamples, size=nSamples)

    # duplicate the samples to be processed
    samplesBefore = X_dataset[randomSamplesBefore].copy()
    samplesAfter = X_dataset[randomSamplesAfter].copy()

    # align samplesBefore and samplesAfter with X_dataset to make them ready to be concatenate
    randomBefore = alignRandomSample(X_dataset, samplesBefore, isBefore=True)
    randomAfter = alignRandomSample(X_dataset, samplesAfter, isBefore=False)

    # concatenate
    longSamples = np.hstack([randomBefore[:, :-2], X_dataset, randomAfter[:, 2:]])

    # randomly select the starting point of each sample
    frameStart = TIME_STAMP - leftFrameLength
    frameEnd = TIME_STAMP - (targetFrameLength - rightFrameLength) + TIME_STAMP

    assert frameStart >= (TIME_STAMP - targetFrameLength //2), "cannot assign more than half of the frame to the first randomly selected sample"
    assert (TIME_STAMP*3 - frameEnd) > targetFrameLength //2 , "cannot assign more than half of the frame to the last randomly selected sample"


    startingFrame = np.random.randint(frameStart, frameEnd, size=nSamples)
    startingPos = startingFrame * 2
    endingPos = startingPos + targetFrameLength * 2

    # chop out the piece of interest
    X_list = []
    for xid, (start, end) in enumerate(zip(startingPos, endingPos)):
        x_sample = longSamples[xid, :]
        X_list.append(x_sample[None, start:end])
    
    X_dataset = np.vstack(X_list)
    return X_dataset
