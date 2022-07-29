from src import pca_for_embeddings
from src import fisher_vector
from src.model_learning import ELM, CascadedNormalizer
from src.data_loading import DataLoader
from src import model_learning
from src.configuration import Config
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import copy
import numpy as np
from time import perf_counter
import datetime
from sklearn.ensemble import RandomForestClassifier


class Tuning:
    def __init__(self, bert_model: str, pca_components: list, gmm_components: list, elm_c: list, power_norm_a: list,
                 config: Config, with_acoustics: str = "", test_set: str = "devel"):
        self.bert_model = bert_model
        self.with_acoustics = with_acoustics
        self.config = config

        self.pca_grid = pca_components
        self.fv_grid = gmm_components
        self.elm_grid = elm_c
        self.power_a = power_norm_a

        self.best_pca = None
        self.best_gmm = None
        self.best_elm = None
        self.best_a = 0
        self.best_score = 0
        self.test_set = test_set
        self.parameter_df = pd.DataFrame(columns=["pca_components", "gmm_components", "c_elm", "a_power",
                                                  "devel_score"])

    def search(self):
        for i_pca in self.pca_grid:
            pca = pca_for_embeddings.EmbeddingPCA(bert_model=self.bert_model,
                                                  words_or_sentences="unigram",
                                                  pca_components=i_pca,
                                                  overwrite_file=False,
                                                  include_acoustics=self.with_acoustics)  # ACOUSTICS
            self.config.linguistic_pca = i_pca
            pca.pca_fit()
            pca.pca_transform(data_set="train")
            pca.pca_transform(data_set="devel")
            pca.pca_transform(data_set="test")

            for j_gmm in self.fv_grid:
                fv = fisher_vector.FisherVector(bert_model=self.bert_model,
                                                data_extension=f"{i_pca}pca",
                                                gmm_components=j_gmm,
                                                acoustics=self.with_acoustics)
                fv.fit_gmm()
                fv.compute_fv()
                self.config.linguistic_gmm = j_gmm

                print(f"Number of FV components: {fv.size}.")
                print("Training ELM...")
                print()

                self.config.linguistic_llds = f"words_{i_pca}pca_{j_gmm}gmm_fv"
                dl = DataLoader(self.config)
                x_train, x_test, y_train, y_test = dl.construct_feature_set()

                for a_power in self.power_a:
                    a, b = CascadedNormalizer(x_train, x_test, "z", "power", "l2", a_power).normalize()
                    assert not np.array_equal(x_train, a) and not np.array_equal(x_test, b)
                    for k_elm in self.elm_grid:

                        # ml = model_learning.ModelLearning(bert_model=self.bert_model, test=self.test_set,
                        #                                   data_extension=f"words{self.with_acoustics}_{i_pca}pca_{j_gmm}gmm_fv", c=k_elm)
                        # ml.normalize_data(feature_level="z", value_level="power", instance_level="l2", a=a_power)

                        model = ELM(c=k_elm)
                        model.fit(a, y_train)
                        pred_probs = model.predict(b)
                        pred = np.argmax(pred_probs, axis=1)

                        if self.test_set != "test":
                            score = recall_score(y_test, pred, average="macro")
                            score = round(score, 4)

                            self.parameter_df.loc[len(self.parameter_df)] = [i_pca, j_gmm, k_elm, a_power, score]
                            print(f"pca: {i_pca}, gmm: {j_gmm}, c_elm: {k_elm}, a_power: {a_power}, devel: {score}")
                            if score > self.best_score:
                                self.best_pca = i_pca
                                self.best_gmm = j_gmm
                                self.best_elm = k_elm
                                self.best_a = a_power
                                self.best_score = score
                        else:
                            pass
                print()


def word_feature_tuning(bert_model, with_acoustics, config, test="devel"):
    tune = Tuning(bert_model=bert_model,
                  pca_components=[2500],
                  gmm_components=[32, 64],
                  elm_c=[0.5, 1.0, 2.0, 4.0],
                  power_norm_a=[-1, 0.4, 0.5, 0.8],  # use -1 for no power normalization
                  config=config_4,
                  with_acoustics=with_acoustics,
                  test_set=test
                  )
    tune.search()

    if test != "test":
        print(f"Best parameters: pca: {tune.best_pca}, gmm: {tune.best_gmm}, "
              f"c_elm: {tune.best_elm}, a_power: {tune.best_a}")
        print(f"Devel score: {tune.best_score * 100}% UAR.")

        tune.parameter_df.to_csv(f"data/results_logs_csv/{bert_model}_words{with_acoustics}_log.csv", mode="a",
                                 header=False, index=False)


# def acoustic_functional_tuning(datatype: str):
#     c = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0]
#     a = [-1, 0.4, 0.5, 0.6, 0.8]
#
#     parameter_df = pd.DataFrame(columns=["c_elm", "a_power", "devel_score"])
#     best_c = 0
#     best_a = 0
#     best_score = 0
#     for c_elm in c:
#         for a_power in a:
#             ml = model_learning.ModelLearning(bert_model="", test="devel", data_extension="", c=c_elm,
#                                               acoustic_functionals=True, function_type=datatype)
#             ml.normalize_data(feature_level="z", value_level="power", instance_level="l2", a=a_power)
#             pred = ml.fit_predict()
#             score = recall_score(ml.y_test, pred, average="macro")
#             score = round(score, 4)
#             parameter_df.loc[len(parameter_df)] = [c_elm, a_power, score]
#             print(f"c_elm: {c_elm}, a_power: {a_power}, devel: {score}")
#             if score > best_score:
#                 best_c = c_elm
#                 best_a = a_power
#                 best_score = score
#     print(f"Best parameters: c_elm: {best_c}, a_power: {best_a}")
#     print(f"Devel score: {best_score * 100}% UAR.")
#     parameter_df.to_csv(f"data/results_logs_csv/acoustic_{datatype}functionals_log.csv", mode="a", header=False, index=False)

def pfi_range_search_fl(config: Config, pfi_l_range: list, pfi_a_range: list):
    best_l_pfi = 0
    best_a_pfi = 0
    best_score = 0
    for linguistic_pfi in pfi_l_range:
        for acoustic_pfi in pfi_a_range:
            config.linguistic_pfi = linguistic_pfi
            config.acoustic_pfi = acoustic_pfi
            pred_probs, y_test = feature_level_fusion(config)
            pred = np.argmax(pred_probs, axis=1)
            score = recall_score(y_test, pred, average="macro")
            score = round(score, 4)
            print(score)

            if score > best_score:
                best_l_pfi = linguistic_pfi
                best_a_pfi = acoustic_pfi
                best_score = score

    print(f"Found best score of {best_score}, with l_pfi={best_l_pfi} and a_pfi={best_a_pfi}")


def pfi_range_search_sl(config_1: Config, config_2: Config, pfi_l_range_1: list, pfi_a_range_1: list,
                        pfi_l_range_2: list, pfi_a_range_2):
    best_l_pfi_1 = 0
    best_a_pfi_1 = 0
    best_l_pfi_2 = 0
    best_a_pfi_2 = 0
    best_score = 0

    parameter_df = pd.DataFrame(columns=["linguistic_1", "acoustic_1", "linguistic_2", "acoustic_2", "devel_score"])

    for linguistic_pfi_1 in pfi_l_range_1:
        config_1.linguistic_pfi = linguistic_pfi_1
        for acoustic_pfi_1 in pfi_a_range_1:
            config_1.acoustic_pfi = acoustic_pfi_1
            for linguistic_pfi_2 in pfi_l_range_2:
                config_2.linguistic_pfi = linguistic_pfi_2
                for acoustic_pfi_2 in pfi_a_range_2:
                    config_2.acoustic_pfi = acoustic_pfi_2

                    score = rf_score_level_fusion(config_pipe_1=config_1, config_pipe_2=config_2)
                    score = round(score, 4)
                    print(score)

                    parameter_df.loc[len(parameter_df)] = [linguistic_pfi_1, acoustic_pfi_1, linguistic_pfi_2,
                                                           acoustic_pfi_2, score]

                    if score > best_score:
                        best_l_pfi_1 = linguistic_pfi_1
                        best_a_pfi_1 = acoustic_pfi_1
                        best_l_pfi_2 = linguistic_pfi_2
                        best_a_pfi_2 = acoustic_pfi_2
                        best_score = score

    print(
        f"Found best score of {best_score}, with l_pfi_1={best_l_pfi_1}, a_pfi_2={best_a_pfi_1}, l_pfi_2={best_l_pfi_2} and a_pfi_2={best_a_pfi_2}")
    parameter_df.to_csv(f"data/results_logs_csv/pfi_scorefusion_log.csv", header=True, index=False)


def feature_level_fusion(config: Config):
    c = config.elm_c
    power_norm_gamma = config.power_norm_gamma

    print("loading data")
    dl = DataLoader(config)
    # dl = DataLoader(train_set=config.train_set, test_set=config.test_set, ling_model="bert",
    #                linguistic_utt=linguistic_features,
    #                acoustic_utt=acoustic_features, utt_functionals=functionals, pca_comp=pca_comp, nr_to_remove=nr_to_remove)
    x_train, x_test, y_train, y_test = dl.construct_feature_set()

    print(f"feature size: {x_train.shape[1]}")

    parameter_df = pd.DataFrame(columns=["c_elm", "a_power", "devel_score"])
    best_c = 0
    best_a = 0
    best_score = 0
    best_pred_probs = None
    best_pred = 0
    for a_power in power_norm_gamma:
        print("normalizing data")
        a, b = CascadedNormalizer(x_train, x_test, "z", "power", "l2", a_power).normalize()
        assert not np.array_equal(x_train, a) and not np.array_equal(x_test, b)
        print("training elm")
        # TODO Change RBF
        #model = ELM(a, b, kernel="rbf")
        model = ELM(a, b)
        for c_elm in c:
            model.fit(a, y_train, c_elm)
            pred_probs = model.predict(b)
            pred = np.argmax(pred_probs, axis=1)

            score = recall_score(y_test, pred, average="macro")
            # TODO ACCURACY
            #score = accuracy_score(y_test, pred)
            # if config.overwrite_pfi:
            #     if config.linguistic_llds:
            #         pfi = model_learning.PFI(model, b, y_test, score, config.linguistic_gmm, config.linguistic_pca)
            #         pfi.feature_performance()
            #         pfi.update_data(f"data/embeddings_pickle/bert_train_{config.linguistic_llds}.pickle")
            #         pfi.update_data(f"data/embeddings_pickle/bert_devel_{config.linguistic_llds}.pickle")
            #         pfi.update_data(f"data/embeddings_pickle/bert_test_{config.linguistic_llds}.pickle")
            #     if config.acoustic_llds:
            #         pfi = model_learning.PFI(model, b, y_test, score, config.acoustic_gmm, config.acoustic_pca)
            #         pfi.feature_performance()
            #         pfi.update_data(f"data/embeddings_pickle/acoustic_train_{config.acoustic_llds}.pickle")
            #         pfi.update_data(f"data/embeddings_pickle/acoustic_devel_{config.acoustic_llds}.pickle")
            #         pfi.update_data(f"data/embeddings_pickle/acoustic_test_{config.acoustic_llds}.pickle")

            score = round(score, 4)
            parameter_df.loc[len(parameter_df)] = [c_elm, a_power, score]
            print(f"c_elm: {c_elm}, a_power: {a_power}, devel: {score}")
            if score > best_score:
                best_c = c_elm
                best_a = a_power
                best_score = score
                best_pred_probs = pred_probs
                best_pred = pred
    print(f"Best parameters: c_elm: {best_c}, a_power: {best_a}")
    print(f"Devel score: {best_score * 100}% UAR.")
    parameter_df.to_csv(
        f"data/results_logs_csv/feature_fusion_{config.acoustic_llds}_{config.linguistic_llds}_{config.acoustic_functionals}_log.csv",
        header=True, index=False)

    return best_pred_probs, y_test, pred


# def testset_feature_level_fusion(elm_c: float, power_g: float, linguistic_features: str = "",
#                                  acoustic_features: str = "", functionals: str = ""):
#     print("loading data")
#     dl = DataLoader(train_set="train_devel", test_set="devel", ling_model="bert",
#                     linguistic_utt=linguistic_features,
#                     acoustic_utt=acoustic_features, utt_functionals=functionals)
#     x_train, x_test, y_train, y_test = dl.construct_feature_set()
#     print(f"feature size: {x_train.shape[1]}")
#     print("normalizing data")
#     a, b = CascadedNormalizer(x_train, x_test, "z", "power", "l2", power_g).normalize()
#     model = ELM(c=elm_c)
#     print("training elm")
#     model.fit(a, y_train)
#     pred_probs = model.predict(b)
#     pred = np.argmax(pred_probs, axis=1)
#
#     score = recall_score(y_test, pred, average="macro")
#     print(round(score, 4))


def fv_encoding(ling_model, pca_comp, acoustic_type, gmm_comp, stride=1, overwrite=False):
    pca = pca_for_embeddings.EmbeddingPCA(bert_model=ling_model,
                                          words_or_sentences="words",
                                          pca_components=pca_comp,
                                          overwrite_file=overwrite,
                                          include_acoustics=acoustic_type)  # ACOUSTICS
    pca.pca_fit(subsamples=stride)
    pca.pca_transform(data_set="train")
    pca.pca_transform(data_set="devel")
    pca.pca_transform(data_set="test")
    fv = fisher_vector.FisherVector(bert_model=ling_model,
                                    data_extension=f"{pca_comp}pca",
                                    gmm_components=gmm_comp,
                                    acoustics=acoustic_type)
    fv.fit_gmm(overwrite=overwrite, subsamples=stride)
    fv.compute_fv()


def score_fusion(config_pipe_1: Config, config_pipe_2: Config, nr_distributions=1000):
    pred_probs_1, y_true_devel, pred_label_1 = feature_level_fusion(config_pipe_1)
    pred_probs_2, y_true_devel, pred_label_2 = feature_level_fusion(config_pipe_2)

    # TODO EDIT FOR NORMAL AND CLASS WEIGHTED CHOOSING
    pred = model_learning.class_weighted_score_fusion(y_true_devel, pred_probs_1, pred_probs_2, nr_distributions)
    # pred = model_learning.score_fusion(y_true_devel, pred_probs_1, pred_probs_2)

    return pred


    # TEST SET METHOD
    # fused_pred = ml2.test_score_fusion(ml1.pred_probs, ml2.pred_probs, 0.5)
    # test_labels = ml2.extract_labels(test)
    # print(recall_score(test_labels, fused_pred, average="macro"))

def save_pred(pred):
    file_name = input("Save predictions as: ")
    file_loc = f"data/labels_csv/{file_name}.pickle"
    with open(file_loc, "wb") as f:
        pickle.dump(pred, f)


def test_set_score_fusion(config_pipe_1: Config, config_pipe_2: Config, gamma: float):
    train_set = "train_devel"
    test_set = "test"
    config_pipe_1.train_set = train_set
    config_pipe_1.test_set = test_set
    config_pipe_2.train_set = train_set
    config_pipe_2.test_set = test_set

    pred_probs_1, y_true_test, pred_label_1 = feature_level_fusion(config_pipe_1)
    pred_probs_2, y_true_test, pred_label_2 = feature_level_fusion(config_pipe_2)

    # TODO EDIT FOR NORMAL AND CLASS WEIGHTED CHOOSING
    # pred = model_learning.test_score_fusion(y_true_test, pred_probs_1, pred_probs_2, gamma)
    pred = model_learning.test_class_weighted_score_fusion(y_true_test, pred_probs_1, pred_probs_2)

    return pred     


def rf_score_level_fusion(config_pipe_1: Config, config_pipe_2: Config):
    pred_devel_1, y_true_devel, p = feature_level_fusion(config_pipe_1)
    pred_devel_2, y_true_devel, p = feature_level_fusion(config_pipe_2)
    confidence_metrics_devel = pd.read_csv(f"data/features_csv/confidence_metrics_{config_pipe_1.test_set}.csv", header=None).values
    devel_data = np.concatenate((pred_devel_1, pred_devel_2, confidence_metrics_devel), axis=1)
    devel_data = np.concatenate((pred_devel_1, pred_devel_2), axis=1)

    clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, oob_score=True)
    clf.fit(devel_data, y_true_devel)

    pred_oob = np.argmax(clf.oob_decision_function_, axis=1)
    recall_oob = recall_score(y_true_devel, pred_oob, average="macro")
    print(f"OOB UAR prediction: {recall_oob}")

    train_set = "train_devel"
    test_set = "test"
    config_pipe_1.train_set = train_set
    config_pipe_1.test_set = test_set
    config_pipe_2.train_set = train_set
    config_pipe_2.test_set = test_set
    pred_test_1, y_true_test, p = feature_level_fusion(config_pipe_1)
    pred_test_2, y_true_test, p = feature_level_fusion(config_pipe_2)
    confidence_metrics_test = pd.read_csv("data/features_csv/confidence_metrics_test.csv", header=None).values
    test_data = np.concatenate((pred_test_1, pred_test_2, confidence_metrics_test), axis=1)
    test_data = np.concatenate((pred_test_1, pred_test_2), axis=1)

    pred = clf.predict(test_data)
    print(recall_score(y_true_test, pred, average="macro"))
    # np.savetxt("data/labels_csv/test_prediction.csv", pred, delimiter=",")
    return recall_oob


def randomforest_for_lda(config):
    dl = DataLoader(config)
    x_train, x_test, y_train, y_test = dl.construct_feature_set()
    print(f"feature size: {x_train.shape[1]}")
    clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = recall_score(y_test, pred, average="macro")
    print(f"recall score is: {round(score, 4)}")


if __name__ == "__main__":
    c_elm = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    power_gamma = [-1, 0.4, 0.5]
    # linguistic_feature_tuning()
    # acoustic_functional_tuning()

    # with_acoustics= None / "_means" / "_means_sd" / "_means_r" / _llds  where sd is standard deviation and r is range
    # bert = "acoustic" for acoustic only

    # word_feature_tuning(bert_model="bert", with_acoustics="", test="test")
    # word_feature_tuning(bert_model="acoustic", with_acoustics="_llds", test="test")

    # fv_encoding("bert", 400, "", 64, stride=1, overwrite=False)
    # fv_encoding("acoustic", 110, "_llds", 200, stride=4, overwrite=False)
    # fv_encoding("acoustic", 400, "_wav2vec2", 128, stride=1, overwrite=False)

    # feature_level_fusion(c_elm, power_gamma, linguistic_features="", acoustic_features="words_llds_110pca_200gmm_fv", functionals="")
    # pred_probs, y_test = feature_level_fusion(c_elm, power_gamma, linguistic_features="", acoustic_features="words_acoustic_llds_110pca_200gmm_fv", functionals="")
    # testset_feature_level_fusion(2, 0.4, linguistic_features="words_400pca_64gmm_fv", acoustic_features="words_llds_110pca_200gmm_fv", functionals="compare")

    # feature_level_fusion([4], [2], linguistic_features="words_llds_110pca_200gmm_fv")

    # score_fusion()
    # test_set_score_fusion()
    train_set = "train"
    test_set = "devel"
    # train_set = "train_devel"
    #  test_set = "test"
    config_1 = Config(train_set=train_set, test_set=test_set, acoustic_llds="words_wav2vec2_400pca_128gmm_fv",
                      acoustic_pca=400, acoustic_gmm=128, bert_model="bert", linguistic_llds="words_400pca_64gmm_fv",
                      linguistic_pca=400, linguistic_gmm=64, elm_c=[32], power_norm_gamma=[0.5])

    config_2 = Config(train_set=train_set, test_set=test_set, acoustic_llds="words_wav2vec2_400pca_128gmm_fv",
                      acoustic_pca=400, acoustic_gmm=128, elm_c=[8], power_norm_gamma=[0.5])

    config_8 = Config(train_set=train_set, test_set=test_set, acoustic_llds="words_compare_llds_130pca_200gmm_fv",
                      acoustic_pca=400, acoustic_gmm=128, elm_c=[2], power_norm_gamma=[0.5])

    config_3 = Config(train_set=train_set, test_set=test_set, bert_model="bert", linguistic_llds="words_400pca_64gmm_fv",
                      linguistic_pca=400, linguistic_gmm=64, elm_c=[1], power_norm_gamma=[0.5])

    config_4 = Config(train_set=train_set, test_set=test_set, acoustic_llds="words_llds_110pca_200gmm_fv",
                      acoustic_functionals="compare", acoustic_pca=110, acoustic_gmm=200, elm_c=[8], power_norm_gamma=[0.4])

    config_5 = Config(train_set=train_set, test_set=test_set,
                      bert_model="acoustic", linguistic_llds="words_means_sd_400pca_64gmm_fv",
                      linguistic_pca=450, linguistic_gmm=64, elm_c=[0.5], power_norm_gamma=[0.5])

    config_6 = Config(train_set=train_set, test_set=test_set, acoustic_functionals="compare", elm_c=[2], power_norm_gamma=[0.8])

    # pfi_range_search_fl(config_1, pfi_l_range=[0, 10, 20], pfi_a_range=[0])
    # pfi_range_search_sl(config_1, config_2, pfi_l_range_1=[0, 10, 20], pfi_a_range_1=[0, 10, 20], pfi_l_range_2=[0], pfi_a_range_2=[0, 10, 20])
    # score = rf_score_level_fusion(config_3, config_2)
    # pred = score_fusion(config_3, config_2)

    # fv_encoding("acoustic", 400, "_wav2vec2", 64, stride=1, overwrite=False)
    # pred = rf_score_level_fusion(config_1, config_4)
    #feature_level_fusion(config_4)
    pred = score_fusion(config_1, config_4, nr_distributions=10000)
    # pred = test_set_score_fusion(config_1, config_4, 0.45)
    #save_pred(pred)
    #np.savetxt("data/labels_csv/test_prediction.csv", pred, delimiter=",")
    # save_pred(pred)
    # word_feature_tuning(bert_model="bow", with_acoustics="", config=config_5)

    # randomforest_for_lda(config_4)
    # acoustic_functional_tuning("compare")
    # score_fusion(acoustic_features="words_llds_110pca_128gmm_fv", linguistic_features="words_400pca_64gmm_fv", test="devel")

    # save_pred(pred)
