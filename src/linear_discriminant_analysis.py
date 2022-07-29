import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.configuration import Config
from src import nli_tools
import operator
import numpy as np


class LDA:
    def __init__(self, config: Config, n_components):
        assert operator.xor(config.linguistic_gmm != 0, config.acoustic_gmm != 0)

        self.main_model = config.bert_model
        self.file_name = config.linguistic_llds
        self.n_components = n_components

        if config.linguistic_gmm != 0:
            self.num_gmm = config.linguistic_gmm
            self.num_pca = config.linguistic_pca
        else:
            self.num_gmm = config.acoustic_gmm
            self.num_pca = config.acoustic_pca

        self.lda_list = []
        for i in range(self.num_gmm):
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
            self.lda_list.append(lda)

    def fit_transform(self, data, labels):
        self.fit(data, labels)
        transformed_data = self.transform(data)
        return transformed_data

    def all_fit_transform(self):
        train_features = nli_tools.single_pickle_loader(f"{self.main_model}_train_{self.file_name}")
        devel_features = nli_tools.single_pickle_loader(f"{self.main_model}_devel_{self.file_name}")
        test_features = nli_tools.single_pickle_loader(f"{self.main_model}_test_{self.file_name}")
        train_labels = nli_tools.train_label_loader()

        train_transformed = self.fit_transform(train_features, train_labels)
        devel_transformed = self.transform(devel_features)
        test_transformed = self.transform(test_features)

        nli_tools.data_pickler(train_transformed, f"{self.main_model}_train_{self.file_name}_lda")
        nli_tools.data_pickler(devel_transformed, f"{self.main_model}_devel_{self.file_name}_lda")
        nli_tools.data_pickler(test_transformed, f"{self.main_model}_test_{self.file_name}_lda")

    def fit(self, data, labels):
        if nli_tools.file_exists(self.file_name):
            print(f"Using saved pickle file for {self.file_name}")
            self.lda_list = nli_tools.load_process_pickle(self.file_name)
        else:
            print(f"Creating new LDAs for {self.file_name}")
            for gmm_comp in range(self.num_gmm):
                single_comp_data = self.__data_slice(data, gmm_comp)
                self.lda_list[gmm_comp].fit(single_comp_data, labels)
                nli_tools.print_progress(gmm_comp+1, self.num_gmm, "LDA models fitted")
            nli_tools.save_process_pickle(self.lda_list, self.file_name)

    def transform(self, data):
        new_data = []
        for gmm_comp in range(self.num_gmm):
            self.lda_list[gmm_comp]._max_components = self.n_components
            single_comp_data = self.__data_slice(data, gmm_comp)
            single_comp_data = self.lda_list[gmm_comp].transform(single_comp_data)
            new_data.append(single_comp_data)
            nli_tools.print_progress(gmm_comp+1, self.num_gmm, "GMM components transformed")
        print()
        transformed_data = np.concatenate(new_data, axis=1)
        return transformed_data

    def __data_slice(self, data, gmm):
        i_start = 2 * self.num_pca * gmm
        i_stop = i_start + 2 * self.num_pca
        single_comp_data = data[:, i_start:i_stop]
        return single_comp_data


if __name__ == "__main__":
    file_name_main = "words_llds_110pca_200gmm_fv_pfi"
    n_components_main = 1

    os.chdir("..")
    config_main = Config(train_set="train", test_set="devel", bert_model="acoustic",
                         linguistic_llds=file_name_main,
                         linguistic_pca=110, linguistic_gmm=200)
    lda_main = LDA(config_main, n_components_main)
    lda_main.all_fit_transform()
