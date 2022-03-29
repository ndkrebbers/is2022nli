from src import pca_for_embeddings
from src import fisher_vector
from src.model_learning import ELM, CascadedNormalizer, DataLoader
from src import model_learning
from sklearn.metrics import recall_score
import pandas as pd
import copy
import numpy as np
from time import perf_counter
import datetime
from sklearn.ensemble import RandomForestClassifier


class Tuning:
    def __init__(self, bert_model: str, pca_components: list, gmm_components: list, elm_c: list, power_norm_a: list, with_acoustics: str = "", test_set: str="devel"):
        self.bert_model = bert_model
        self.with_acoustics = with_acoustics

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
                                                  words_or_sentences="words",
                                                  pca_components=i_pca,
                                                  overwrite_file=False,
                                                  include_acoustics=self.with_acoustics)  # ACOUSTICS
            pca.pca_fit()
            pca.pca_transform(data_set="train")
            pca.pca_transform(data_set=self.test_set)

            for j_gmm in self.fv_grid:
                fv = fisher_vector.FisherVector(bert_model=self.bert_model,
                                                data_extension=f"{i_pca}pca",
                                                gmm_components=j_gmm,
                                                acoustics=self.with_acoustics)
                fv.fit_gmm()
                fv.compute_fv()

                print(f"Number of FV components: {fv.size}.")
                print("Training ELM...")
                print()

                for k_elm in self.elm_grid:
                    for a_power in self.power_a:
                        ml = model_learning.ModelLearning(bert_model=self.bert_model, test=self.test_set,
                                                          data_extension=f"words{self.with_acoustics}_{i_pca}pca_{j_gmm}gmm_fv", c=k_elm)
                        ml.normalize_data(feature_level="z", value_level="power", instance_level="l2", a=a_power)
                        pred = ml.fit_predict()

                        if self.test_set != "test":
                            score = recall_score(ml.y_test, pred, average="macro")
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


def word_feature_tuning(bert_model, with_acoustics, test="devel"):
    tune = Tuning(bert_model=bert_model,
                  pca_components=[200],# [400, 450, 500],
                  gmm_components=[128],
                  elm_c=[1], #[0.2, 0.5, 1.0, 2.0, 4.0],
                  power_norm_a=[0.5], # [-1, 0.5, 0.6, 0.8],  # use -1 for no power normalization
                  with_acoustics=with_acoustics,
                  test_set=test
                  )
    tune.search()

    if test != "test":
        print(f"Best parameters: pca: {tune.best_pca}, gmm: {tune.best_gmm}, "
              f"c_elm: {tune.best_elm}, a_power: {tune.best_a}")
        print(f"Devel score: {tune.best_score * 100}% UAR.")

        tune.parameter_df.to_csv(f"data/results_logs_csv/{bert_model}_words{with_acoustics}_log.csv", mode="a", header=False, index=False)


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


def feature_level_fusion(elm_c: list, power_g: list, linguistic_features: str = "", acoustic_features: str = "", functionals: str = "", x_train="train", x_test="devel"):
    c = elm_c
    a = power_g

    print("loading data")
    dl = DataLoader(train_set=x_train, test_set=x_test, ling_model="bert",
                    linguistic_utt=linguistic_features,
                    acoustic_utt=acoustic_features, utt_functionals=functionals)
    x_train, x_test, y_train, y_test = dl.construct_feature_set()
    print(f"feature size: {x_train.shape[1]}")

    parameter_df = pd.DataFrame(columns=["c_elm", "a_power", "devel_score"])
    best_c = 0
    best_a = 0
    best_score = 0
    best_pred_probs = None
    for a_power in a:
        print("normalizing data")
        a, b = CascadedNormalizer(x_train, x_test, "z", "power", "l2", a_power).normalize()
        assert not np.array_equal(x_train, a) and not np.array_equal(x_test, b)
        for c_elm in c:
            model = ELM(c=c_elm)
            print("training elm")
            model.fit(a, y_train)
            pred_probs = model.predict(b)
            pred = np.argmax(pred_probs, axis=1)

            score = recall_score(y_test, pred, average="macro")
            score = round(score, 4)
            parameter_df.loc[len(parameter_df)] = [c_elm, a_power, score]
            print(f"c_elm: {c_elm}, a_power: {a_power}, devel: {score}")
            if score > best_score:
                best_c = c_elm
                best_a = a_power
                best_score = score
                best_pred_probs = pred_probs
    print(f"Best parameters: c_elm: {best_c}, a_power: {best_a}")
    print(f"Devel score: {best_score * 100}% UAR.")
    parameter_df.to_csv(f"data/results_logs_csv/feature_fusion_{acoustic_features}_{linguistic_features}_{functionals}_log.csv", header=True, index=False)

    return best_pred_probs, y_test

def testset_feature_level_fusion(elm_c: float, power_g: float, linguistic_features: str = "", acoustic_features: str = "", functionals: str = ""):
    print("loading data")
    dl = DataLoader(train_set="train_devel", test_set="devel", ling_model="bert",
                    linguistic_utt=linguistic_features,
                    acoustic_utt=acoustic_features, utt_functionals=functionals)
    x_train, x_test, y_train, y_test = dl.construct_feature_set()
    print(f"feature size: {x_train.shape[1]}")
    print("normalizing data")
    a, b = CascadedNormalizer(x_train, x_test, "z", "power", "l2", power_g).normalize()
    model = ELM(c=elm_c)
    print("training elm")
    model.fit(a, y_train)
    pred_probs = model.predict(b)
    pred = np.argmax(pred_probs, axis=1)

    score = recall_score(y_test, pred, average="macro")
    print(round(score, 4))


def fv_encoding(ling_model, pca_comp, acoustic_type, gmm_comp, stride=0, overwrite=False):
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

def score_fusion():
    pred_probs_1, y_test_1 = feature_level_fusion([4], [0.4], linguistic_features="", acoustic_features="words_acoustic_llds_110pca_200gmm_fv", functionals="compare")
    print("Prediction using acoustics complete.")

    pred_probs_2, y_test_2 = feature_level_fusion([1], [0.5], linguistic_features="words_400pca_64gmm_fv", acoustic_features="", functionals="")
    print("Prediction using linguistics complete.")

    model_learning.score_fusion(y_test_1, pred_probs_1, pred_probs_2)

    # TEST SET METHOD
    # fused_pred = ml2.test_score_fusion(ml1.pred_probs, ml2.pred_probs, 0.5)
    # test_labels = ml2.extract_labels(test)
    # print(recall_score(test_labels, fused_pred, average="macro"))

def test_set_score_fusion():
    pred_probs_1, y_test_1 = feature_level_fusion([4], [0.4], linguistic_features="",
                                                  acoustic_features="words_llds_110pca_200gmm_fv",
                                                  functionals="compare")
    print("Prediction using acoustics complete.")

    pred_probs_2, y_test_2 = feature_level_fusion([1], [0.5], linguistic_features="words_400pca_64gmm_fv",
                                                  acoustic_features="", functionals="")
    print("Prediction using linguistics complete.")

    fused_pred = model_learning.test_score_fusion(pred_probs_1, pred_probs_2, 0.55)
    print(recall_score(y_test_1, fused_pred, average="macro"))


def rf_score_level_fusion():
    pred_devel_1, y_true_devel = feature_level_fusion([1], [0.5], linguistic_features="words_400pca_64gmm_fv",
                                                      acoustic_features="words_wav2vec2_400pca_128gmm_fv")
    pred_devel_2, y_true_devel = feature_level_fusion([4], [0.4], acoustic_features="words_acoustic_llds_110pca_200gmm_fv",
                                                      functionals="compare")
    confidence_metrics_devel = pd.read_csv("data/features_csv/confidence_metrics_devel.csv", header=None).values
    devel_data = np.concatenate((pred_devel_1, pred_devel_2, confidence_metrics_devel), axis=1)

    clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, oob_score=True)
    clf.fit(devel_data, y_true_devel)

    pred_oob = np.argmax(clf.oob_decision_function_, axis=1)
    recall_oob = recall_score(y_true_devel, pred_oob, average="macro")
    print(f"OOB UAR prediction: {recall_oob}")

    pred_test_1, y_true_test = feature_level_fusion([1], [0.5], linguistic_features="words_400pca_64gmm_fv",
                                                    acoustic_features="",
                                                    x_train="train_devel", x_test="test")
    pred_test_2, y_true_test = feature_level_fusion([4], [0.4], acoustic_features="words_acoustic_llds_110pca_200gmm_fv",
                                                    functionals="compare", x_train="train_devel", x_test="test")
    confidence_metrics_test = pd.read_csv("data/features_csv/confidence_metrics_test.csv", header=None).values
    test_data = np.concatenate((pred_test_1, pred_test_2, confidence_metrics_test), axis=1)

    pred = clf.predict(test_data)
    print(recall_score(y_true_test, pred, average="macro"))
    np.savetxt("data/labels_csv/test_prediction.csv", pred, delimiter=",")

if __name__ == "__main__":
    c_elm = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    power_gamma = [-1, 0.5]
    # linguistic_feature_tuning()
    # acoustic_functional_tuning()

    # with_acoustics= None / "_means" / "_means_sd" / "_means_r" / _llds  where sd is standard deviation and r is range
    # bert = "acoustic" for acoustic only

    # word_feature_tuning(bert_model="bert", with_acoustics="", test="test")
    # word_feature_tuning(bert_model="acoustic", with_acoustics="_llds", test="test")

    # fv_encoding("acoustic", 110, "_llds", 200, stride=4, overwrite=True)

    # feature_level_fusion(c_elm, power_gamma, linguistic_features="", acoustic_features="words_llds_110pca_200gmm_fv", functionals="")
    # pred_probs, y_test = feature_level_fusion(c_elm, power_gamma, linguistic_features="", acoustic_features="words_acoustic_llds_110pca_200gmm_fv", functionals="")
    # testset_feature_level_fusion(2, 0.4, linguistic_features="words_400pca_64gmm_fv", acoustic_features="words_llds_110pca_200gmm_fv", functionals="compare")

    # feature_level_fusion([4], [2], linguistic_features="words_llds_110pca_200gmm_fv")

    #score_fusion()
    # test_set_score_fusion()

    rf_score_level_fusion()


    # acoustic_functional_tuning("compare")
    # score_fusion(acoustic_features="words_llds_110pca_128gmm_fv", linguistic_features="words_400pca_64gmm_fv", test="devel")

