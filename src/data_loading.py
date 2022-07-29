import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.configuration import Config


class DataLoader:
    def __init__(self, config: Config):
        # def __init__(self, train_set: str, test_set: str, ling_model: str = "", linguistic_utt: str = "",
        #              acoustic_utt: str = "", utt_functionals: str = "", pca_comp: int = 0, nr_to_remove: int = 0):
        #     self.train_set = config.train_set
        #     self.test_set = config.test_set
        #
        #     self.pca_comp = pca_comp
        #     self.nr_to_remove = nr_to_remove
        self.config = config

        # self.linguistic_utt = config.linguistic_llds
        # self.acoustic_utt = config.acoustic_llds
        # self.utt_functionals = config.acoustic_functionals
        assert self.config.linguistic_llds or self.config.acoustic_llds or self.config.acoustic_functionals or \
               self.config.linguistic_functionals, "There is no data to extract."

        self.label_encoder = preprocessing.LabelEncoder()
        if self.config.acoustic_functionals == "compare":
            self.selected_cols = self.__select_5300()

    def construct_feature_set(self):
        if self.config.train_set == "train_devel":
            x_train = self.__get_features("train")
            x_devel = self.__get_features("devel")
            x_train = np.concatenate((x_train, x_devel), axis=0)

            y_train = self.__read_labels("train")
            y_devel = self.__read_labels("devel")
            y_train = np.concatenate((y_train, y_devel), axis=0)
        else:
            x_train = self.__get_features(self.config.train_set)
            y_train = self.__read_labels(self.config.train_set)

        x_test = self.__get_features(self.config.test_set)
        y_test = self.__read_labels(self.config.test_set)
        return x_train, x_test, y_train, y_test

    def __get_features(self, t_d_t: str) -> np.array:
        nr_of_rows = self.__determine_size(t_d_t)
        features = np.empty((nr_of_rows, 0))

        if self.config.linguistic_llds:
            ling_f = self.__read_fv_features(self.config.bert_model, t_d_t, self.config.linguistic_llds)

            # if self.config.old_order:
            #     ling_f = self.revert_order(ling_f)
            #     arr_len = ling_f.shape[1]
            #     remove_features = self.config.linguistic_pca * self.config.linguistic_pfi
            #
            #     ling_f = ling_f[:, :arr_len - remove_features]
            #     ling_f = np.concatenate((ling_f[:, :int(0.5*arr_len)-remove_features], ling_f[:, int(0.5*arr_len):]), axis=1)
            #
            # else:
            ling_f = ling_f[:, :ling_f.shape[1] - (2 * self.config.linguistic_pca * self.config.linguistic_pfi)]  # REMOVE worst features
            features = np.concatenate((features, ling_f), axis=1)
        if self.config.acoustic_llds:
            acou_f = self.__read_fv_features("acoustic", t_d_t, self.config.acoustic_llds)
            # if self.config.old_order:
            #     acou_f = self.revert_order(acou_f)
            #     arr_len = acou_f.shape[1]
            #     remove_features = self.config.acoustic_pca * self.config.acoustic_pfi
            #
            #     acou_f = acou_f[:, :arr_len - remove_features]
            #     acou_f = np.concatenate((acou_f[:, :int(0.5 * arr_len) - remove_features], acou_f[:, int(0.5*arr_len):]), axis=1)
            #
            # else:
            acou_f = acou_f[:, :acou_f.shape[1] - (2 * self.config.acoustic_pca * self.config.acoustic_pfi)]
            features = np.concatenate((features, acou_f), axis=1)
        if self.config.acoustic_functionals:
            func_f = self.__read_utt_functionals(t_d_t)
            features = np.concatenate((features, func_f), axis=1)

        if self.config.linguistic_functionals:
            linguistic_functional_features = self.__read_linguistic_functionals(t_d_t)
            features = np.concatenate((features, linguistic_functional_features), axis=1)

        return features


    @staticmethod
    def revert_order(fv):
        for i in range(fv.shape[0]):
            temp = np.reshape(fv[i], (2, int(0.5*len(fv[i]))), order="F")
            fv[i] = temp.flatten()
        return fv

    @staticmethod
    def __read_fv_features(model: str, t_d_t: str, specs: str) -> np.array:
        file_loc = f"data/embeddings_pickle/{model}_{t_d_t}_{specs}.pickle"
        with open(file_loc, "rb") as f:
            features = pickle.load(f)
        return features

    def __read_utt_functionals(self, t_d_t: str) -> np.array:
        if self.config.acoustic_functionals == "compare":
            file_loc = f"data/features_csv/{self.config.acoustic_functionals}_{t_d_t}.csv"
            df = pd.read_csv(file_loc)
            features = df.values
            features = features[:, self.selected_cols]
        else:
            df = pd.read_csv("data/features_csv/mfcc_rastaplpc_10functionals.csv", header=None)  # Slightly inefficient
            df = df.drop(df.columns[0], axis=1)
            if t_d_t == "train":
                features = df.tail(3300)
            elif t_d_t == "devel":
                features = df.head(965)
            else:  # t_d_t == "test"
                features = df.iloc[965:965 + 867, ]
        return features

    def __read_labels(self, t_d_t: str) -> np.array:
        file_loc = f"data/labels_csv/{t_d_t}_labels.csv"
        if t_d_t != "test":
            df = pd.read_csv(file_loc)
        else:
            df = pd.read_csv(file_loc, delimiter=";")
        labels = df["L1"].values

        if t_d_t == "train":
            labels = self.label_encoder.fit_transform(labels)
        else:
            labels = self.label_encoder.transform(labels)
        return labels

    @staticmethod
    def __determine_size(t_d_t):
        if t_d_t == "train":
            size = 3300
        elif t_d_t == "devel":
            size = 965
        elif t_d_t == "train_devel":
            size = 3300 + 965
        elif t_d_t == "test":
            size = 867
        else:
            size = 0
        assert size != 0
        return size

    @staticmethod
    def __select_5300():
        cols = pd.read_csv("data/features_csv/feature_ranking_5300.csv", header=None).values
        cols = cols.transpose()[0]
        return cols

    def __read_linguistic_functionals(self, t_d_t):
        file_loc = f"data/embeddings_pickle/bert_{t_d_t}_bow.pickle"
        # file_loc = f"data/embeddings_pickle/sentencebert_{t_d_t}_separate.pickle"
        with open(file_loc, "rb") as f:
            features = pickle.load(f)

        # means_and_sds = []
        # for par in features:
        #     mean_par = np.mean(par, axis=0)
        #     sd_par = np.std(par, axis=0)
        #     means_and_sds.append(np.concatenate((mean_par, sd_par)))
        # features = np.array(means_and_sds)



        return features