import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""

General method for preprocessing any dataset in the best way posible in a generalized way.

The steps done are:
1. Fix missing values
2. Normalize categorical data
3. Normalize all data to the same numerical domain

"""


def preprocess(dataset):
    dataset = fix_na(dataset)
    dataset = fix_categorical(dataset)
    dataset = normalize(dataset)
    return dataset


"""

Fixes missing values for all the columns of the given DataFrame.

"""


def fix_na(dataset):
    for column in dataset:
        # Get not None data to check its type
        i = 0
        while dataset[column][i] is None:
            i = i+1

        # Check type
        # If it's float the we compute mean and we assign it to NA values
        if isinstance(dataset[column][i], float):
            col_mean = dataset[column].mean(skipna=True)
            dataset[column] = dataset[column].fillna(col_mean)
        else:
            col_moda = dataset[column].mode()
            dataset[column] = dataset[column].fillna(col_moda[0])
    return dataset


"""

Normalizes categorical data by separating categories into different columns of 0s and 1s.

"""


def fix_categorical(dataset):
    dataset = pd.get_dummies(dataset, dtype=float)
    return dataset


"""

Normalizes dataset by putting all numeric data into same scale of values (domain of values).

"""


def normalize(dataset):
    scaler = MinMaxScaler()
    dataset[list(dataset.columns.values)] = scaler.fit_transform(dataset)
    return dataset
