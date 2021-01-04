import pandas as pd
import numpy as np


class PreProcessing():
    def winGenerator(self, X, sequences_length):
        # 自己建立窗口
        # init
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        # split windows
        index = np.arange(0, data.shape[0] - (sequences_length - 1), 1, int)
        sequences = [data[i:i + sequences_length] for i in index]
        print(len(sequences))
        return sequences


