"""Utils functions."""

import os

import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder as LE


def create_directory(directory_path: str):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    directory_path : str
       The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except FileExistsError:
            raise FileExistsError("Already exists.")


def load_data_aeon(file_name):
    """Load the time series data through aeon.

    Parameters
    ----------
    file_name: str,
        The name of the dataset to load.

    Returns
    -------
    xtrain: np.ndarray, shape (n_samples, length_TS, n_channels)
        The training samples.
    ytrain: np.ndarray, shape (n_samples,)
        The training labels.
    xtest: np.ndarray, shape (m_samples, length_TS, n_channels)
        The testing samples.
    ytest: np.ndarray, shape (m_samples)
        The testing labels.
    """
    xtrain, ytrain = load_classification(name=file_name, split="train")
    xtest, ytest = load_classification(name=file_name, split="test")

    ytrain, ytest = _encode_labels(ytrain, ytest)

    xtrain = np.swapaxes(xtrain, axis1=1, axis2=2)
    xtest = np.swapaxes(xtest, axis1=1, axis2=2)

    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    return xtrain, ytrain, xtest, ytest


def load_data_local(file_name):
    """Load the time series data.

    Parameters
    ----------
    file_name: str,
        The name of the dataset to load.

    Returns
    -------
    xtrain: np.ndarray, shape (n_samples, length_TS, n_channels)
        The training samples.
    ytrain: np.ndarray, shape (n_samples,)
        The training labels.
    xtest: np.ndarray, shape (m_samples, length_TS, n_channels)
        The testing samples.
    ytest: np.ndarray, shape (m_samples)
        The testing labels.
    """
    folder_path = "/home/aismailfawaz/datasets/TSC/UCRArchive_2018/"
    folder_path += file_name + "/"

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if os.path.exists(test_path) <= 0:
        raise FileNotFoundError("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    ytrain = _encode_labels(ytrain)
    ytest = _encode_labels(ytest)

    xtrain = np.expand_dims(xtrain, axis=-1)
    xtest = np.expand_dims(xtest, axis=-1)

    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    return xtrain, ytrain, xtest, ytest


def znormalisation(x):
    """Z-normalize the time series data.

    Parameters
    ----------
    x: np.ndarray, shape (n_samples, length_TS, n_channels)
        The input time series set.

    Returns
    -------
    x: np.ndarray, shape (n_samples, length_TS, n_channels)
        The normalzied version of the set.
    """
    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def _encode_labels(ytrain, ytest):
    labenc = LE()

    ytrain_encoded = labenc.fit_transform(ytrain)
    ytest_encoded = labenc.transform(ytest)

    return ytrain_encoded, ytest_encoded
