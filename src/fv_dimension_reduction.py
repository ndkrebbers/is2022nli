import os
import nli_tools
from src.configuration import Config
import operator
from sklearn.decomposition import PCA
import numpy as np


class PCAfv:
    def __init__(self, config: Config, component_percentage):
        assert operator.xor(config.linguistic_gmm != 0, config.acoustic_gmm != 0)

        self.main_model = config.bert_model
        self.file_name = config.linguistic_llds
        self.component_percentage = component_percentage

        if config.linguistic_gmm != 0:
            self.num_gmm = config.linguistic_gmm
            self.num_pca = config.linguistic_pca
        else:
            self.num_gmm = config.acoustic_gmm
            self.num_pca = config.acoustic_pca

        self.pca_list = []
        for i in range(self.num_gmm):
            pca = PCA(random_state=42)
            self.pca_list.append(pca)

    def all_fit_transform(self):
        train_features = nli_tools.single_pickle_loader(f"{self.main_model}_train_{self.file_name}")
        devel_features = nli_tools.single_pickle_loader(f"{self.main_model}_devel_{self.file_name}")
        test_features = nli_tools.single_pickle_loader(f"{self.main_model}_test_{self.file_name}")

        train_transformed = self.fit_transform(train_features)
        devel_transformed = self.transform(devel_features)
        test_transformed = self.transform(test_features)

        nli_tools.data_pickler(train_transformed, f"{self.main_model}_train_{self.file_name}_reduced_pca")
        nli_tools.data_pickler(devel_transformed, f"{self.main_model}_devel_{self.file_name}_reduced_pca")
        nli_tools.data_pickler(test_transformed, f"{self.main_model}_test_{self.file_name}_reduced_pca")

    def fit_transform(self, data):
        self.fit(data)
        transformed_data = self.transform(data)
        return transformed_data

    def fit(self, data):
        if nli_tools.file_exists(self.file_name+"reduced_pca"):
            print(f"Using saved pickle file for {self.file_name}")
            self.pca_list = nli_tools.load_process_pickle(self.file_name+"reduced_pca")
        else:
            print(f"Creating new PCAs for {self.file_name}")
            for gmm_comp in range(self.num_gmm):
                single_comp_data = self.__data_slice(data, gmm_comp)
                self.pca_list[gmm_comp].fit(single_comp_data)
                nli_tools.print_progress(gmm_comp+1, self.num_gmm, "PCA models fitted")
            nli_tools.save_process_pickle(self.pca_list, self.file_name+"reduced_pca")

    def transform(self, data):
        new_data = []
        sum_explained_var = []
        for gmm_comp in range(self.num_gmm):
            # self.pca_list[gmm_comp]._max_components = self.n_components
            single_comp_data = self.__data_slice(data, gmm_comp)
            single_comp_data = self.pca_list[gmm_comp].transform(single_comp_data)
            fraction = round(single_comp_data.shape[1] * (self.component_percentage / 100))
            single_comp_data = single_comp_data[:, : fraction]
            sum_explained_var.append(sum(self.pca_list[gmm_comp].explained_variance_ratio_[: fraction]))
            new_data.append(single_comp_data)
            nli_tools.print_progress(gmm_comp+1, self.num_gmm, "GMM components transformed")
        print()
        print(sum(sum_explained_var) / self.num_gmm)
        transformed_data = np.concatenate(new_data, axis=1)
        return transformed_data

    def __data_slice(self, data, gmm):
        i_start = 2 * self.num_pca * gmm
        i_stop = i_start + 2 * self.num_pca
        single_comp_data = data[:, i_start:i_stop]
        return single_comp_data


if __name__ == "__main__":
    file_name_main = "words_llds_110pca_200gmm_fv_pfi"
    os.chdir("..")
    config_main = Config(train_set="train", test_set="devel", bert_model="acoustic",
                         linguistic_llds=file_name_main,
                         linguistic_pca=110, linguistic_gmm=200)
    pca_main = PCAfv(config_main, component_percentage=75)
    pca_main.all_fit_transform()