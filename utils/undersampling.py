import pandas as pd
import numpy as np


def delete_random(df, column, value, percent):
    np.random.seed(0)
    mask = df[column] == value
    indices = df[mask].index
    remove_n = int(percent * len(indices))
    drop_indices = np.random.choice(indices, remove_n, replace=False)
    df = df.drop(drop_indices)
    return df


def count_zeroes(df):
    return (df['label']==0.0).value_counts()[True]


train = pd.read_csv('./data/train.csv')

print('Before : ', count_zeroes(train))
dropped = delete_random(train, 'label', 0.0, 0.5)
print('After : ', count_zeroes(dropped))

dropped.to_csv('./data/dropped_train.csv', sep=',', encoding='utf-8', index=False)
