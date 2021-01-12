import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from DeepADoTS_master.src.evaluation import Evaluator


class UPDATE:

    def update(self,  X: pd.DataFrame, predict, label, updateCount):
        load_model = torch.load('E:\\ML\\BA\\phase\\phas_' + updateCount - 1 + '\\2DimWithOneSudden.pkl')
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + 30] for i in range(data.shape[0] - 30 + 1)]
        load_model.fit(X, sequences)
        torch.save(load_model, 'E:\\ML\\BA\\phase\\phase_' + updateCount + '\\2DimWithOneSudden.pkl')
        score, error, output = load_model.predict(predict, update=True)
        evaluate = Evaluator()
        t = evaluate.get_optimal_threshold(y_test=label[2000:], score=score)
        print(t)
        return t, score, error, output


