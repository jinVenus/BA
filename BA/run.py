from BA.config.DataPreprocessing import PreProcessing
import json

import numpy as np
import pandas as pd
import torch
from DeepADoTS_master.src.evaluation import Evaluator
from BA.algorithms.lstm_enc_dec_axl_2 import LSTMED
from BA.config.ImageMaker import DrawFigure
from BA.algorithms.update import UPDATE

# read config
with open("/DeepADoTS_master/src/algorithms/paramter.json", 'r') as load_f:
    load_dict = json.load(load_f)

# load the dataset
data = pd.read_csv("/BA/datasets/Synthetic/2DimWithOneSudden/2DimWithOneSudden.csv", sep=",", header=None)

data.interpolate(inplace=True)
data.bfill(inplace=True)
data = data.values

# init dataSplitPoint
trainPoint = 6000
validationPoint = 7000
endPoint = data.size

data.columns = [['x1', 'x2', 'x3', 'outcome']]
print(data)
label = data[['outcome']]

data_train = data.drop(columns=['outcome'])
data_train = (data_train - data_train.mean(axis=0)) / data_train.std(axis=0)

# split train/ validation/ test set
train = data_train[:trainPoint]
validation = data_train[trainPoint:validationPoint]
prediction = data_train[validationPoint:]

# split windows
trainSequences = PreProcessing.winGenerator(train, load_dict["sequence_length"])
validationSequences = PreProcessing.winGenerator(validation, load_dict["sequence_length"])
predictionSequences = PreProcessing.winGenerator(prediction, load_dict["sequence_length"])

# init model
model = LSTMED(num_epochs=load_dict["num_epochs"], batch_size=load_dict["batch_size"],
               hidden_size=load_dict["hidden_size"], sequence_length=load_dict["sequence_length"])

# train model
model.fit(train, trainSequences)

# save model
torch.save(model, 'E:\\ML\\BA\\phase\\phase_0\\2DimWithOneSudden.pkl')

# validation
score, error, output = model.predict(validation, validationSequences)

# compute t
evaluate = Evaluator(data_train[validationPoint:], trainPoint, validationPoint)
t = evaluate.get_optimal_threshold(y_test=label[trainPoint:validationPoint], score=score)
print(t)

# init error
scores3 = []
outputs3 = []
errors3 = []

# predictionWithStream
new_label = np.zeros(3000)
buffer = []

threshold = []
upd = UPDATE()
updateCount = 0
for j in range(prediction.shape[0] - 20*load_dict["sequence_length"] + 1):
    model = torch.load('update.pkl')
    if ((j + 20*load_dict["sequence_length"]) > prediction.shape[0] - 20*load_dict["sequence_length"] + 1):
        data = prediction[j:]
        index = np.arange(0, data.shape[0] - load_dict["sequence_length"], 30, int)
        sequences = [prediction[i:i + load_dict["sequence_length"]] for i in index]

        win_score, win_error, win_output = model.predict(prediction, seq=sequences, update=False, t=t, data=prediction)
        win_score = pd.DataFrame(win_score)
        win_error = pd.DataFrame(win_error)
        win_output = pd.DataFrame(win_output)

        scores3.append(win_score)
        outputs3.append(win_output)
        errors3.append(win_error)

        for id, score in enumerate(win_score):
            if score >= 0.75 * t:
                buffer.append(prediction[j+id])
                # if score > t:
                #     new_label[j+id] = 1
            if len(buffer) >= 3000:
                updateCount += 1
                buffer = pd.DataFrame(buffer)
                for i in range(load_dict["sequence_length"]):
                    new_label[1970 + i] = 1
                t = upd.update(buffer[:2000], buffer, new_label, updateCount)
                threshold.append(t)
                buffer = []
                new_label = np.zeros(3000)

        j += 20*load_dict["sequence_length"]

    else:
        data = prediction[j:j+20*load_dict["sequence_length"]]
        index = np.arange(0, data.shape[0] - load_dict["sequence_length"] + 1, load_dict["sequence_length"], int)
        sequences = [data[i:i + 30] for i in index]

        win_score, win_error, win_output = model.predict(prediction, seq=sequences, update=False, t=t, data=prediction)
        win_score = pd.DataFrame(win_score)
        win_error = pd.DataFrame(win_error)
        win_output = pd.DataFrame(win_output)

        scores3.append(win_score)
        outputs3.append(win_output)
        errors3.append(win_error)

        for id, score in enumerate(win_score):
            if score >= 0.75 * t:
                buffer.append(prediction[j:j+id])
                # if win_score > t:
                #     new_label[j+id] = 1
            if len(buffer) >= 3000:
                updateCount += 1
                buffer = pd.DataFrame(buffer)
                for i in range(30):
                    new_label[1970 + i] = 1
                t = upd.update(buffer[:2000], buffer, new_label, updateCount)
                threshold.append(t)
                buffer = []
                new_label = np.zeros(3000)
        j += 20*load_dict["sequence_length"]



print(threshold)


# compute Score
from DeepADoTS_master.src.evaluation import Evaluator
computeScore = Evaluator(load_dict["sequence_length"])
score, error, output = Evaluator.get_optimal_threshold(prediction, scores3, errors3, outputs3)

# draw Result
DrawFigure.drawScore(scores3, "ScoreBefore")
DrawFigure.drawReconstructionError(errors3)
DrawFigure.drawThreshold(scores3, t)

from BA.config.Evaluation import ComputeEvalution

ComputeEvalution.evalution(scores3, t, label, load_dict, trainPoint, validationPoint)

print("update times : ")
print(updateCount)