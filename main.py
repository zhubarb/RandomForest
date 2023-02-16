# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import zipfile, kaggle
import pathlib

class Decision_Tree:

    def __init__(self, max_leaf_nodes, cats, conts):
        self.avaliable_cols = cats + conts
        self.splits = []

    def get_best_split_col(d):
        best_split_val = 1e10
        best_split_col = None
        for k, v in d.items():
            split_val = v[1]
            if split_val < best_split_val:
                best_split_val = split_val
                best_split_col = k
        return (k, v)

    def min_col(self, df, nm):
        col, y = df[nm], df[dep]
        unq = col.dropna().unique()
        scores = np.array([self.score(col, y, o) for o in unq if not np.isnan(o)])
        idx = scores.argmin()
        return unq[idx], scores[idx]

    def score(self, col, y, split):
        lhs = col <= split
        return (self._side_score(lhs, y) + self._side_score(~lhs, y)) / len(y)

    def _side_score(side, y):
        tot = side.sum()
        if tot <= 1: return 0
        return y[side].std() * tot

    def fit(self, df, available_cols, selected_cols):
        if len(available_cols) <= 1:  # leaf node
            print('Training ended')
            return self.min_col(df, available_cols)
        else:  # mid node
            d = {o: self.min_col(df, o) for o in available_cols}
            col, [thresh, val] = self.get_best_split_col(d)
            self.splits += (col, [thresh, val])
            print(self.splits)
            # Update Available features and already selected ones
            selected_cols += col
            available_cols.remove(col)
            # Split the data in two
            lhs = df.loc[df[col] < thresh]
            rhs = df.loc[df[col] >= thresh]
            # Fit both hand-sides of the split dataset
            self.fit(lhs, available_cols, selected_cols)
            self.fit(rhs, available_cols, selected_cols)
        # Press the green button in the gutter to run the script.


def proc_data(df, modes):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)


if __name__ == '__main__':

    path = pathlib.Path('./titanic')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

    df = pd.read_csv(path / 'train.csv')
    tst_df = pd.read_csv(path / 'test.csv')

    modes = df.mode().iloc[0]
    proc_data(df, modes)
    proc_data(tst_df, modes)

    cats = ["Sex", "Embarked"]
    conts = ['Age', 'SibSp', 'Parch', 'LogFare', "Pclass"]
    dep = "Survived"
