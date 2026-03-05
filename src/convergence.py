import pandas as pd
import numpy as np
from pathlib import Path


class ConvergenceTracker:

    def __init__(self, maximize=False):
        self.maximize = maximize
        self.scores = []
        self.params = []

    def log(self, score, params=None):
        self.scores.append(score)
        self.params.append(params)

    def dataframe(self):

        df = pd.DataFrame({
            "iteration": np.arange(len(self.scores)),
            "score": self.scores
        })

        if self.maximize:
            df["best_so_far"] = np.maximum.accumulate(df["score"])
        else:
            df["best_so_far"] = np.minimum.accumulate(df["score"])

        return df

    def save(self, path):

        df = self.dataframe()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)