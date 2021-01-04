import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class DrawFigure:
    def __init__(self, load_dict):
        self.load_dict = load_dict
        now = datetime.now()
        self.date_time = now.strftime("%m%d%Y")

    def drawThreshold(self, score, t):
        plt.figure(figsize=(28, 5))  # 图像大小
        x = np.arange(len(score))
        y = score
        plt.plot(x, y)
        plt.title("Threshold")
        plt.ylabel('Value')
        plt.xlabel('Data')
        plt.axhline(y=t, color='r', linestyle='-')
        plt.savefig(
            "E:\\ML\\BA\\result\\" + self.date_time + "\\Synthetic\\" + self.load_dict[
                "dataName"] + "\\Threshold.png",
            format='png')
        plt.show()
        plt.show()

    def drawReconstructionError(self, error):
        plt.figure(figsize=(28, 5))  # 图像大小
        x = np.arange(len(error))
        y = error
        plt.plot(x, y)
        plt.title("ReconstructionError")
        plt.ylabel('ReconstructionError')
        plt.xlabel('Data')
        plt.savefig(
            "E:\\ML\\BA\\result\\" + self.date_time + "\\Synthetic\\" + self.load_dict[
                "dataName"] + "\\ReconstructionError.png",
            format='png')
        plt.show()

    def drawScore(self, score, figName="Score"):
        plt.figure(figsize=(28, 5))  # 图像大小
        x = np.arange(len(score))
        y = score
        plt.plot(x, y)
        plt.title("Score")
        plt.ylabel('Score')
        plt.xlabel('Data')
        plt.savefig(
            "E:\\ML\\BA\\result\\" + self.date_time + "\\Synthetic\\" + self.load_dict[
                "dataName"] + "\\" +figName+ ".png",
            format='png')
        plt.show()