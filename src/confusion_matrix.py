import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import os


class CMatrix:
    def __init__(self, y_true, y_pred):
        self.le = LabelEncoder()
        self.le.fit(y_true)
        y_pred = y_pred.astype(int).squeeze(1)
        y_pred = self.le.inverse_transform(y_pred)
        self.labels = [self.le.classes_[3], self.le.classes_[2], self.le.classes_[5],
                       self.le.classes_[8], self.le.classes_[0], self.le.classes_[10],
                       self.le.classes_[4], self.le.classes_[9], self.le.classes_[6],
                       self.le.classes_[7], self.le.classes_[1]]
        self.cm = confusion_matrix(y_true, y_pred, labels=self.labels)

    def plot(self):
        data = self.cm / self.cm.astype(np.float).sum(axis=1)[:, np.newaxis] * 100
        hm = sn.heatmap(data, annot=True, cmap="Blues", fmt='.1f', xticklabels=self.labels,
                        yticklabels=self.labels, cbar=False)  # Remove lognorm if not exercise 5
        hm.xaxis.tick_top()
        hm.xaxis.set_label_position('top')
        plt.ylabel("True label")
        plt.xlabel("Prediction label")
        plt.yticks(rotation=0)
        plt.savefig("cm.png")
        plt.show()


if __name__ == "__main__":
    os.chdir("..")
    y_true_df = pd.read_csv(f"data/labels_csv/test_labels.csv", delimiter=";")
    y_t = y_true_df["L1"].values
    y_pred_df = pd.read_csv(f"data/labels_csv/test_prediction.csv", header=None)
    y_p = y_pred_df.values

    cm = CMatrix(y_t, y_p)
    cm.plot()
