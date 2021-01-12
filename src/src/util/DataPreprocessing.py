import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal

"""
1. preprocess data: missing value, scaling, map label column to {0, 1}, ....
2. split data into
    filter only normal data for sn, ...
    init
        sn, vn1,vn2, ...
    online
        stream.csv
3. save data into corresponding folder
"""


def PreProcessing(data_path, sequences_length, experiment_folder, init_data_portion, training_data_rate,
                  validation_data_rate):
    data = pd.read_csv(data_path, sep=",", header=None)
    data = pd.DataFrame(data)
    data.columns = [['x1', 'x2', 'x3', 'outcome']]
    print(data)
    label = data[['outcome']]
    data_train = data.drop(columns=['outcome'])
    data_train = (data_train - data_train.mean(axis=0)) / data_train.std(axis=0)

    # init dataSplitPoint
    initPoint = int(data.size * init_data_portion)
    trainPoint = int(data.size * init_data_portion * training_data_rate)
    validationPoint = int(trainPoint + data.size * init_data_portion * validation_data_rate)

    # split train/ validation/ test set
    train = data_train[:trainPoint]
    validation = data_train[trainPoint:validationPoint]
    prediction = data_train[validationPoint:initPoint]
    stream_data = data_train[initPoint:]

    # split windows
    trainSequences = splitWindow(train, sequences_length, label, train)
    validationSequences = splitWindow(validation, sequences_length, label, train)
    predictionSequences = splitWindow(prediction, sequences_length, label, train)

    return trainSequences, validationSequences, predictionSequences, train, validation, prediction, data_train, label, stream_data, initPoint, trainPoint, validationPoint


def splitWindow(data, sequences_length, label, train):
    data.interpolate(inplace=True)
    data.bfill(inplace=True)
    data = data.values

    train.interpolate(inplace=True)
    train.bfill(inplace=True)
    train = train.values

    # split windows
    index = np.arange(0, data.shape[0] - sequences_length - 1, 1)
    sequences = [data[i:i + sequences_length] for i in index]

    if np.array_equal(data, train):

        sequences = []
        for i in range(data.shape[0] - sequences_length + 1):
            sequences.append(data[i:i + sequences_length])

        index = []
        i = 0
        length = data.size
        for l in label[:length]:
            if l == 1:
                for j in range(sequences_length):
                    index.append(i + j)

            i = i + 1
        print(index)
        index = np.unique(index)

        np.set_printoptions(threshold=np.inf)
        for i in reversed(index):
            sequences = sequences[:i] + sequences[i + 1:]

    return sequences
