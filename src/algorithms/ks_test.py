from scipy import stats
import pandas as pd
import numpy as np


class kstest:
    def __init__(self, alpha):
        self.alpha = alpha
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

    def test(self, win_hist, win_new):
        '''
        st: the ks distance
        p_value: if p_value <= alpha, drift is detected
        '''
        st, p_value = stats.ks_2samp(win_hist, win_new)  # historical window and new window can have different sizes
        return st, p_value

    def detect_drift(self, p_value):
        return p_value <= self.alpha


def ksTest(test):
    hist = []  # keep all his
    new = []
    count = 0
    index = []
    distance = []
    i = 0
    ind = np.arange(1000, 100001, 100)
    ks = kstest(alpha=0.01)
    for idx, row in test.iterrows():
        #     date = row['dateRep']
        hist.append(row[0])
        if len(new) <= 200:
            new.append(row[0])
        else:
            new = new[50:] + [row[0]]
        if len(hist) >= ind[i]:
            st, pvalue = ks.test(hist, new)
            #         print(' KS-statistic: {round(st, 2)}, p-value: {pvalue}')

            i += 1
            if pvalue < 0.01 and st >= 0.5:
                print(idx, st, pvalue)
                index.append(idx)
                distance.append(st)
                count += 1
                if count == 10:
                    return index, distance

            else:
                count = 0

    return 0, 0
