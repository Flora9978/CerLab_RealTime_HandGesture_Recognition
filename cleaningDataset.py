import numpy as np


def loadDataset(datasetFileList, badSampleIdxFiles, TIME_STEPS, DIMENSION, teleportation_threshold):
    """
    Load multiple sets of data from datasetFileList and remove badSamples if 
    badSampleIdxFiles is specified.

    output:
        X_dataset: ndarray of shape (samples x timestempx2)
        y_dataset: 1-d array of shape (samples, )
    """
    X_list = []
    y_list = []

    if badSampleIdxFiles is None:
        badSampleIdxFiles = [None] * len(datasetFileList)

    for csvfileName, badsampleIdxFile in zip(datasetFileList, badSampleIdxFiles):
        X_dataset = np.loadtxt(csvfileName, delimiter=',', dtype='float32', usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)))
        y_dataset = np.loadtxt(csvfileName, delimiter=',', dtype='int32', usecols=(0))

        X_dataset, y_dataset = filterBadData(X_dataset, y_dataset, badsampleIdxFile, teleportation_threshold) # comment if you want to keep mutation 
        X_list.append(X_dataset)
        y_list.append(y_dataset)

    X_dataset = np.vstack(X_list)
    y_dataset = np.concatenate(y_list)

    return X_dataset, y_dataset

def filterBadData(X_dataset, y_dataset, badsampleIdxFile=None, teleportation_threshold=0.2):
    """
    Using the index in a badSampleIdxListTXT file to filter out bad samples in
    X_dataset and y_dataset.

    inputs:
        X_dataset: ndarray of shape (samples x timestempx2)
        y_dataset: 1-d array of shape (samples, )
        badsampleIdxFile: string
        teleportation_threshold: int, allowed maximum teleportation distance

    
    outputs:
        X_dataset: ndarray, filtered y_dataset
        y_dataset: ndarray, filtered y_dataset

    """

    print("original shape = ", X_dataset.shape)
    TIME_STEPS = X_dataset.shape[1] // 2
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

    if badsampleIdxFile:
        badSampleIdxList = np.loadtxt(badsampleIdxFile)
        filt = np.ones(len(X_dataset), dtype=bool)
        for idx in badSampleIdxList:
            idx = int(idx)
            filt[idx] = False

        X_dataset = X_dataset[filt]
        y_dataset = y_dataset[filt]

    return X_dataset, y_dataset