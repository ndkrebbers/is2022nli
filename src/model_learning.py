from typing import Tuple
import os

# import src.elm_kernel as elm
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.metrics import recall_score, accuracy_score
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise
# import statistics
from src.configuration import Config
from src.data_loading import DataLoader

from numpy.random import default_rng
from src import nli_tools
import scipy.stats as stats


class ELM:
    def __init__(self, x_train, x_test, weighted=False, kernel='linear'):
        super(self.__class__, self).__init__()

        assert kernel in ["rbf", "linear"]
        self.x_train = []

        self.weighted = weighted
        self.beta = []
        self.kernel = kernel
        self.kernel_func_train = self.kernel_train(x_train)
        self.kernel_func_test = self.kernel_test(x_train, x_test)

    def kernel_train(self, x_train):
        print(self.kernel)
        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_train)
        else:  # kernel == linear
            kernel_func = pairwise.linear_kernel(x_train)
        return kernel_func

    def kernel_test(self, x_train, x_test):
        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_test, x_train)
        else:  # kernel == linear
            kernel_func = pairwise.linear_kernel(x_test, x_train)
        return kernel_func

    def fit(self, x_train, y_train, c):
        """
        Calculate beta using kernel.
        :param x_train: features of train set
        :param y_train: labels of train set
        :return:
        """
        self.x_train = x_train
        class_num = max(y_train) + 1
        n = len(x_train)
        y_one_hot = np.eye(class_num)[y_train]

        if self.weighted:
            W = np.zeros((n, n))
            hist = np.zeros(class_num)
            for label in y_train:
                hist[label] += 1
            hist = 1 / hist
            for i in range(len(y_train)):
                W[i, i] = hist[y_train[i]]
            beta = np.matmul(np.linalg.inv(np.matmul(W, self.kernel_func_train) +
                                           np.identity(n) / c), np.matmul(W, y_one_hot))
        else:
            beta = np.matmul(np.linalg.inv(
                self.kernel_func_train + np.identity(n) / c), y_one_hot)
        self.beta = beta

    def predict(self, x_test):
        """
        Predict label probabilities of new data using calculated beta.
        :param x_test: features of new data
        :return: class probabilities of new data
        """
        # if self.kernel == 'rbf':
        #    kernel_func = pairwise.rbf_kernel(x_test, self.x_train)
        # else:  # kernel == linear
        #    kernel_func = pairwise.linear_kernel(x_test, self.x_train)
        pred = np.matmul(self.kernel_func_test, self.beta)
        return pred

    # def normalize_data(self, feature_level: str = None, value_level: str = None,
    #                    instance_level: str = None, a: float = 1):
    #     """
    #     Cascaded normalization of data.
    #     :param feature_level: normalization over columns
    #     :param value_level: all data cells
    #     :param instance_level: normalization over rows
    #     :param a: tuning parameter for power normalization
    #     :return:
    #     """
    #     if feature_level == "z":
    #         scaler = preprocessing.StandardScaler()
    #         scaler.fit(self.x_train)
    #         self.x_train = scaler.transform(self.x_train)
    #         self.x_test = scaler.transform(self.x_test)
    #
    #     if value_level == "power":
    #         if a != -1:
    #             self.x_train = self.power_norm(self.x_train, a)
    #             self.x_test = self.power_norm(self.x_test, a)
    #
    #     if instance_level == "l2":
    #         self.x_train = preprocessing.normalize(self.x_train, norm=instance_level)
    #         self.x_test = preprocessing.normalize(self.x_test, norm=instance_level)
    #
    # @staticmethod
    # def power_norm(z, a):
    #     """
    #     Power normalization of float.
    #     :param z: float
    #     :param a: optimization parameter
    #     :return: normalized float
    #     """
    #     return np.multiply(np.sign(z), np.power(np.abs(z), a))


def class_weighted_score_fusion(y_true: list, cp_a: np.array, cp_b: np.array, nr_distributions=1000):
    nr_distributions = nr_distributions
    print(f"Using {nr_distributions} uniform distributions.")

    lower, upper = 0, 1
    mu, sigma = 0.5, 0.1
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    np.random.seed(42)
    weights = X.rvs((nr_distributions, 11))

    # rng = default_rng(42)
    # weights = rng.uniform(low=0.0, high=1.0, size=(nr_distributions, 11))
    weights = weights.round(3)

    best_uar = 0
    best_dist = []
    best_pred = []

    for dist in weights:
        arr = np.tile(dist, (cp_a.shape[0], 1))
        neg_dist = np.subtract(1, dist)
        neg_ar = np.tile(neg_dist, (cp_a.shape[0], 1))
        fused_confidences = np.multiply(arr, cp_a) + np.multiply(neg_ar, cp_b)
        y_pred = fused_confidences.argmax(axis=-1)
        uar = recall_score(y_true, y_pred, average="macro")
        # TODO CHANGE BACK
        # uar = accuracy_score(y_true, y_pred)
        if uar > best_uar:
            best_uar = uar
            best_dist = dist
            best_pred = y_pred

    print(
        f"Best score of {round(best_uar, 4) * 100}%  UAR is found using dist={best_dist}.")
    nli_tools.save_process_pickle(best_dist, "best_distribution")
    return best_pred

def test_class_weighted_score_fusion(y_true, cp_a: np.array, cp_b: np.array):
    dist = nli_tools.load_process_pickle("best_distribution")
    arr = np.tile(dist, (cp_a.shape[0], 1))
    neg_dist = np.subtract(1, dist)
    neg_ar = np.tile(neg_dist, (cp_a.shape[0], 1))
    fused_confidences = np.multiply(arr, cp_a) + np.multiply(neg_ar, cp_b)
    y_pred = fused_confidences.argmax(axis=-1)
    uar = recall_score(y_true, y_pred, average="macro")
    print(
        f"Best score of {round(uar, 4) * 100}%  UAR is found using dist={dist}.")

    return fused_confidences.argmax(axis=-1)


def score_fusion(y_true: list, cp_a: np.array, cp_b: np.array):
    best_uar = 0
    best_gamma = 0
    best_pred = []
    for gamma in range(0, 105, 5):
        gamma = gamma / 100
        fused_confidences = gamma * cp_a + (1 - gamma) * cp_b
        y_pred = fused_confidences.argmax(axis=-1)
        uar = recall_score(y_true, y_pred, average="macro")
        # TODO CHANGE BACK
        # uar = accuracy_score(y_true, y_pred)
        if uar > best_uar:
            best_uar = uar
            best_gamma = gamma
            best_pred = y_pred

    print(
        f"Best score of {round(best_uar, 4) * 100}%  UAR is found using gamma={best_gamma}.")
    return best_pred


def test_score_fusion(y_true, cp_a: np.array, cp_b: np.array, gamma):
    fused_confidences = gamma * cp_a + (1 - gamma) * cp_b
    y_pred = fused_confidences.argmax(axis=-1)
    uar = recall_score(y_true, y_pred, average="macro")
    print(
        f"Score fusion using gamma={gamma} gives {round(uar, 4) * 100}% UAR.")
    return fused_confidences.argmax(axis=-1)


# def svm_scorefusion(y_true: list, cp_a: np.array, cp_b: np.array, y_true_train, cp_a_train, cp_b_train):
#     # data = np.concatenate((cp_a_train, cp_b_train), axis=1)
#     # clf = SVC()
#     # clf.fit(data, y_true_train)
#     #
#     # test_data = np.concatenate((cp_a, cp_b), axis=1)
#     # y_pred = clf.predict(test_data)
#     # uar = recall_score(y_true, y_pred, average="macro")
#
#     data = np.concatenate((cp_a, cp_b), axis=1)
#     clf = SVC()
#     scores = cross_val_score(clf, data, y_true, cv=5)
#     print(statistics.mean(scores))


class PFI:
    def __init__(self, elm, x_dev, y_dev, old_score, gmm_components, pca_comp):
        self.model = elm
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.gmm_comp = gmm_components
        self.pca_comp = pca_comp
        self.old_score = old_score
        self.order = []

    def feature_performance(self):
        feature_importances = []
        for comp in range(self.gmm_comp):
            mutated_set = self.shuffle_featureset(comp)
            score = self.calculate_score(mutated_set)
            print(score)
            feature_importances.append((comp, score))

        feature_dict = dict(feature_importances)
        features_ordered = [k for k, v in sorted(
            feature_dict.items(), key=lambda item: item[1])]

        self.order = features_ordered

    def update_data(self, data_loc):
        with open(data_loc+".pickle", "rb") as f:
            data = pickle.load(f)
        organized_data = self.re_organize_data(data)
        with open(data_loc+"_pfi.pickle", "wb") as f:
            pickle.dump(organized_data, f)

    def re_organize_data(self, dataset):
        organized_data = []
        for comp in self.order:
            start_i = 2 * comp * self.pca_comp
            stop_i = start_i + 2 * self.pca_comp
            organized_data.append(dataset[:, start_i:stop_i])
        organized_data = np.concatenate(organized_data, axis=1)
        return organized_data

    def shuffle_featureset(self, gmm_comp):
        arr = self.x_dev.copy()
        start_i = 2 * self.pca_comp * gmm_comp
        stop_i = start_i + 2 * self.pca_comp
        np.random.seed(42)

        np.apply_along_axis(np.random.shuffle, 0, arr[:, start_i:stop_i])
        assert not np.array_equal(arr, self.x_dev)
        return arr

    def calculate_score(self, x_mutated):
        pred_probss = self.model.predict(x_mutated)
        preddd = np.argmax(pred_probss, axis=1)
        new_score = recall_score(self.y_dev, preddd, average="macro")
        score_dif = new_score - self.old_score
        return score_dif  # larger negative difference indicates more importance of the feature


class CascadedNormalizer:
    def __init__(self, x_train: np.array, x_test: np.array, feature_lvl: str = None, value_lvl: str = None,
                 instance_lvl: str = None, power_gamma: float = 1):
        self.x_train = x_train
        self.x_test = x_test

        self.feature_lvl = feature_lvl
        self.value_lvl = value_lvl
        self.instance_lvl = instance_lvl

        self.power_gamma = power_gamma

    def normalize(self) -> Tuple[np.array, np.array]:
        self.__feature_norm()
        self.__value_norm()
        self.__instance_norm()
        return self.x_train, self.x_test

    def __feature_norm(self):
        if self.feature_lvl == "z":
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.x_train)
            self.x_train = scaler.transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
        else:
            pass

    def __value_norm(self):
        if self.value_lvl == "power" and self.power_gamma != -1:
            self.x_train = np.multiply(np.sign(self.x_train), np.power(
                np.abs(self.x_train), self.power_gamma))
            self.x_test = np.multiply(np.sign(self.x_test), np.power(
                np.abs(self.x_test), self.power_gamma))
        else:
            pass

    def __instance_norm(self):
        if self.instance_lvl == "l2":
            self.x_train = preprocessing.normalize(
                self.x_train, norm=self.instance_lvl)
            self.x_test = preprocessing.normalize(
                self.x_test, norm=self.instance_lvl)
        else:
            pass


if __name__ == "__main__":
    os.chdir("..")
    llds = "words_wav2vec2_400pca_128gmm_fv"
    bert_model = "acoustic"
    config_pipe_1_m = Config(train_set="train", test_set="devel", acoustic_llds=llds,
                             acoustic_functionals="", acoustic_pca=400, acoustic_gmm=128, acoustic_pfi=0, bert_model=bert_model,
                             linguistic_llds="", linguistic_functionals="", linguistic_pca=400,
                             linguistic_gmm=128, linguistic_pfi=0, overwrite_pca=False, overwrite_gmm=False,
                             overwrite_pfi=False, elm_c=[1], power_norm_gamma=[0.5])
    dl_m = DataLoader(config_pipe_1_m)
    a, b, cc, d = dl_m.construct_feature_set()
    a, b = CascadedNormalizer(a, b, "z", "power", "l2", 0.5).normalize()
    modell = ELM(c=1)
    modell.fit(a, cc)

    pred_probs = modell.predict(b)
    predd = np.argmax(pred_probs, axis=1)
    pred_score = recall_score(d, predd, average="macro")
    print(pred_score)

    pfi = PFI(modell, b, d, pred_score, config_pipe_1_m.linguistic_gmm,
              config_pipe_1_m.linguistic_pca)
    pfi.feature_performance()
    pfi.update_data(f"data/embeddings_pickle/{bert_model}_train_{llds}")
    pfi.update_data(f"data/embeddings_pickle/{bert_model}_devel_{llds}")
    pfi.update_data(f"data/embeddings_pickle/{bert_model}_test_{llds}")

    # ml = ModelLearning(bert_model="sbert", test="devel", data_extension="sentences", c=0.8, acoustic_functionals=True)
    # ml.normalize_data(feature_level="z", value_level="power", instance_level="l2", a=0.6)
    # pred = ml.fit_predict()

    # print(recall_score(ml.y_test, pred, average="macro"))
    # score_fusion([0, 0, 0, 0, 0, 0, 0, 0], np.random.rand(8, 4), np.random.rand(8, 4))
