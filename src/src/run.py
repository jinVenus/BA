from BA.src.util.DataPreprocessing import PreProcessing
from BA.src.util.model_loader import ModelLoader
import json
import os
import itertools

import numpy as np
import pandas as pd
import torch
from DeepADoTS_master.src.evaluation import Evaluator
from BA.src.algorithms.lstm_enc_dec_axl import LSTMED
from BA.src.configuration.ImageMaker import DrawFigure
from BA.src.algorithms.online_learner import update, online_learning

"""
1. define parameters
2. call initialization
3. call online learner
"""

# -----------------

root = 'exp_2021_01_07'

datasets = ['agots_sudden']

# models = ['rnn_ae', 'lstm_ae', 'dagmm', 'gru_ae', 'vanilla_ae']
models = ['lstm_ae']
config_path = os.path.join('configuration', 'config.json')

hyperparameters = {
    # 'smtp': {
    #     "learning_rate" : 0.01,
    #     "hidden_size": [10, 40, 50, 80],
    #     "sequence_length": [30, 50, 80],
    #     "init_data_portion": 0.3,
    #     "training_data_rate": 0.6,
    #     "validation_data_rate": 0.3
    # },
    'agots_sudden': {
        # "learning_rate": [0.01],
        "hidden_size": [80],
        "sequence_length": [30],
        "init_data_portion": 0.3,
        "training_data_rate": 0.6,
        "validation_data_rate": 0.3
    }
}
# ------------------

# read config
# default
with open(config_path, 'r') as load_f:
    config_dict = json.load(load_f)  # rename as config_dict

for dataset in datasets:
    for model in models:
        data_path = 'E:\\ML\\BA\\data\\2DimWithOneSudden\\2DimWithOneSudden.csv'

        for hidden_size, sequence_length, init_data_portion, training_data_rate, validation_data_rate in itertools.product(
                # hyperparameters[]['learning_rate'],
                hyperparameters[dataset]['hidden_size'],
                hyperparameters[dataset]['sequence_length'],
                hyperparameters[dataset]['init_data_portion'],
                hyperparameters[dataset]['training_data_rate'],
                hyperparameters[dataset]['validation_data_rate']):
            # hidden_size = 80
            # sequence_length = 30
            # init_data_portion = 0.3
            # training_data_rate = 0.6
            # validation_data_rate = 0.3

            experiment_folder = 'E:\\ML\\BA\\results/{root}\\{model}\\win_{sequence_length}_hs_{hidden_size}_lr_0.01_{dataset}'

            init_sn_path = 'E:\\ML\\BA\\{experiment_folder}\\init\\data\\sn.csv'
            init_vn_path = 'E:\\ML\\BA\\{experiment_folder}\\init\\data\\vn.csv'
            init_tn_path = 'E:\\ML\\BA\\{experiment_folder}\\init\\data\\tn.csv'

            model = ModelLoader(model, config_dict)
            # torch.save(model, f'{experiment_folder}/init.th')
            # torch.save(model, f'{experiment_folder}/online/phase' + 0 + '\\2DimWithOneSudden.th')
            # initialization
            sn, vn, tn, train, validation, prediction, data, label, stream_data, initPoint, trainPoint, validationPoint = PreProcessing(data_path, sequence_length,
                                                                                                    experiment_folder,
                                                                                                    init_data_portion,
                                                                                                    training_data_rate,
                                                                                                    validation_data_rate)

            # vn = pd.DataFrame(vn)
            # sn = pd.DataFrame(sn)
            # tn = pd.DataFrame(tn)
            # vn.to_csv(init_vn_path, index=False, sep=',')
            # sn.to_csv(init_sn_path, index=False, sep=',')
            # tn.to_csv(init_tn_path, index=False, sep=',')

            mu, sigma = model.fit(train, seq=sn)
            scores, errors, outputs = model.predict(validation, update=False, seq=vn, data=validation)

            evaluate = Evaluator()
            t = evaluate.get_optimal_threshold(
                y_test=label[trainPoint:validationPoint], score=scores)

            print(t)

            # mu.to_csv('E:\\ML\\BA\\{experiment_folder}\\init\\parameters\\mu.csv', index=False, sep=',')
            # sigma.to_csv('E:\\ML\\BA\\{experiment_folder}\\init\\parameters\\sigma.csv', index=False, sep=',')

            online_learning(model, t, stream_data, sequence_length, 20, experiment_folder)
