import sys
import os
import pickle
import pandas as pd
from sklearn import preprocessing


def path_finder(main_model: str = "", specification: str = "", pca: int = 0, gmm: int = 0, pfi: bool = False):
    elements = []
    if main_model:
        elements.append(main_model)
    if specification:
        elements.append(specification)
    if pca > 0:
        elements.append(str(pca)+"pca")
    if gmm > 0:
        elements.append(str(gmm)+"gmm_fv")
    if pfi:
        elements.append("pfi")
    base_path = "_".join(elements)
    assert base_path, "Path is empty"

    train = base_path + "_train.pickle"
    devel = base_path + "_devel.pickle"
    test = base_path + "_test.pickle"
    return train, devel, test


def print_progress(current_i, max_i, message):
    sys.stdout.flush()
    sys.stdout.write(f"\r{message}: %i/{max_i}" % current_i)


def single_pickle_loader(file_name):
    file_loc = f"data/embeddings_pickle/{file_name}.pickle"
    with open(file_loc, "rb") as f:
        features = pickle.load(f)
    return features


def data_pickler(obj, file_name):
    file_loc = f"data/embeddings_pickle/{file_name}.pickle"
    with open(file_loc, "wb") as f:
        pickle.dump(obj, f)


def save_process_pickle(obj, file_name):
    file_loc = f"data/process_pickle/{file_name}.pickle"
    with open(file_loc, "wb") as f:
        pickle.dump(obj, f)


def load_process_pickle(file_name):
    file_loc = f"data/process_pickle/{file_name}.pickle"
    with open(file_loc, "rb") as f:
        features = pickle.load(f)
    return features


def train_label_loader():
    label_encoder = preprocessing.LabelEncoder()
    file_loc = f"data/labels_csv/train_labels.csv"
    df = pd.read_csv(file_loc)
    labels = df["L1"].values
    labels = label_encoder.fit_transform(labels)
    return labels


def devel_label_loader():
    label_encoder = preprocessing.LabelEncoder()
    file_loc = f"data/labels_csv/devel_labels.csv"
    df = pd.read_csv(file_loc)
    labels = df["L1"].values
    labels = label_encoder.fit_transform(labels)
    return labels


def test_label_loader():
    label_encoder = preprocessing.LabelEncoder()
    file_loc = f"data/labels_csv/test_labels.csv"
    df = pd.read_csv(file_loc, delimiter=";")
    labels = df["L1"].values
    labels = label_encoder.fit_transform(labels)
    return labels


def file_exists(file_name):
    return os.path.exists(f"data/process_pickle/{file_name}.pickle")

