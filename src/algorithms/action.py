import pandas as pd
import numpy as np


def active(X: pd.DataFrame, index):
    X.append(index)
    return X
