import pandas as pd
import os
import nli_tools
import pickle
from statsmodels.stats.contingency_tables import mcnemar


def mc_nemar(y_true, y_clf_1, y_clf_2, classifiers=""):
    """
    Apply mc nemar statistical test on two classifier predictions
    :param y_true: true labels
    :param y_clf_1: prediction of classifier 1
    :param y_clf_2: prediction of classifier 2
    :param classifiers: string of classifiers to compare
    :return: -
    """
    binarized_answer_1 = [1 if y_clf_1[x] == y_true[x] else 0 for x in range(len(y_true))]
    binarized_answer_2 = [1 if y_clf_2[x] == y_true[x] else 0 for x in range(len(y_true))]

    df = pd.DataFrame(list(zip(binarized_answer_1, binarized_answer_2)), columns=['clf1', 'clf2'])
    contingency_table = pd.crosstab(df["clf1"], df["clf2"])
    assert contingency_table.iloc[0,1] + contingency_table.iloc[1,0] >= 25
    stat_res = mcnemar(contingency_table)


    print(classifiers)
    print(f"{stat_res}\n")


if __name__ == "__main__":
    os.chdir("..")
    y_true_main = nli_tools.test_label_loader()

    pred1 = "rplp_mfcc_fv"
    pred2 = "bert_rplp_mfcc_sf"

    with open(f"data/labels_csv/{pred1}.pickle", "rb") as f:
        labels1 = pickle.load(f)

    with open(f"data/labels_csv/{pred2}.pickle", "rb") as f:
        labels2 = pickle.load(f)

    mc_nemar(y_true_main, labels1, labels2)
