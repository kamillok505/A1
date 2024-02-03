import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, L, n, h, ξ, w, θ, delta, d_w, d_θ):
        self.L = L
        self.n = n
        self.h = h
        self.ξ = ξ
        self.w = w
        self.θ = θ
        self.delta = delta
        self.d_w = d_w
        self.d_θ = d_θ

class Dataset:
    def __init__(self, names, features, patterns, boundary, train, test, train_df, test_df, rangesTrain, rangesTest):
        self.names = names
        self.features = features
        self.patterns = patterns
        self.boundary = boundary
        self.train = train
        self.test = test
        self.train_df = train_df
        self.test_df = test_df
        self.rangesTrain = rangesTrain
        self.rangesTest = rangesTest

def preprocess_file(df):
    if df.isnull().values.any():
        print("Data contains NaN values")
        
    df = df.dropna().copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_ranges(df):
    return [(df[col].min(), df[col].max()) for col in df.columns]

def scale(df, ranges, s_min=0.0, s_max=1.0):
    for col_idx, col in enumerate(df.columns):
        x_min, x_max = ranges[col_idx]
        df[col] = s_min + ((s_max - s_min) / (x_max - x_min)) * (df[col] - x_min)
    return df


def descale(df, ranges, s_min=0.0, s_max=1.0):
    for col_idx, col in enumerate(df.columns):
        x_min, x_max = ranges[col_idx]
        df[col] = x_min + ((x_max - x_min) / (s_max - s_min)) * (df[col] - s_min)
    return df


def data_slicer(path, boundary=0.8, preprocess=False):
    print("...DataSlicer()")
    df = pd.read_csv(path)
    if preprocess:
        df = preprocess_file(df)

    train_df, test_df = train_test_split(df, test_size=1 - boundary)
    ranges_train = get_ranges(train_df)
    ranges_test = get_ranges(test_df)

    scale(train_df, ranges_train)
    scale(test_df, ranges_test)

    print(f"Features: {df.shape[1]} | Patterns: {df.shape[0]} | Boundary: {boundary} | Training: {train_df.shape[0]} | Test: {test_df.shape[0]}")

    return Dataset(df.columns, df.shape[1], df.shape[0], boundary, train_df.to_numpy(), test_df.to_numpy(), train_df, test_df, ranges_train, ranges_test)

def copy_dataset(data):
    return Dataset(data.names, data.features, data.patterns, data.boundary, data.train.copy(), data.test.copy(), data.train_df.copy(), data.test_df.copy(), data.rangesTrain.copy(), data.rangesTest.copy())
