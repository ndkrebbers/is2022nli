import pickle
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
from os import path
from time import perf_counter
import datetime
import os

class EmbeddingPCA:
    def __init__(self, bert_model: str, words_or_sentences: str, pca_components: int, overwrite_file: bool,
                 include_acoustics: str = ""):
        self.bert_model = bert_model
        self.word_or_sent = words_or_sentences
        self.nr_of_pcs = pca_components
        self.pca = PCA(random_state=42)
        self.scaler = preprocessing.StandardScaler()
        self.explained_variance = float
        self.overwrite = overwrite_file
        self.acoustics = include_acoustics

    def pca_fit(self, subsamples=1):
        """
        Create principal components using the training set
        :return:
        """
        if self.bert_model != "acoustic":
            with open(f"data/embeddings_pickle/{self.bert_model}_train_{self.word_or_sent}.pickle", 'rb') as f:
                utterance_arrays = pickle.load(f)

        if self.acoustics != "":
            if self.acoustics == "_wav2vec2":
                with open(f"data/embeddings_pickle/acoustic_train_{self.word_or_sent}{self.acoustics}.pickle", 'rb') as f:
                    utterance_acoustics = pickle.load(f)
            else:
                with open(f"data/acoustics_pickle/train{self.acoustics}.pickle", 'rb') as f:
                    utterance_acoustics = pickle.load(f)

            if self.bert_model != "acoustic":
                for i in range(len(utterance_acoustics)):
                    assert len(utterance_arrays[i]) == len(utterance_acoustics[i])
                utterance_arrays = [np.concatenate((utterance_arrays[x], utterance_acoustics[x]), axis=1) for x in range(len(utterance_arrays))]
            else:
                utterance_arrays = utterance_acoustics

        # utterance_arrays = self.tensor_to_numpy(utterance_tensors)

        utterance_arrays = utterance_arrays
        if self.bert_model == "bow" or self.bert_model == "tfidf":
            all_word_embeddings = utterance_arrays
        else:
            all_word_embeddings = np.concatenate(utterance_arrays, axis=0)

        if self.acoustics == "_acoustic_llds" or self.acoustics == "_compare_llds" or self.acoustics == "_llds" or self.acoustics == "_wav2vec2":
            all_word_embeddings = all_word_embeddings[0::subsamples, ]

        self.scaler = self.scaler.fit(all_word_embeddings)

        file_loc = f"data/process_pickle/pca_{self.bert_model}_train_{self.word_or_sent}{self.acoustics}.pickle"
        if path.exists(file_loc) and not self.overwrite:
            print(f"Using existing pca pickle file for {self.bert_model}_train_"
                  f"{self.word_or_sent}{self.acoustics}_{self.nr_of_pcs}pca.")
            with open(file_loc, "rb") as f:
                pca_pickle = pickle.load(f)
                self.pca = pca_pickle
        else:
            print(f"Fitting new PCA to {self.bert_model}_train_{self.word_or_sent}{self.acoustics}_{subsamples}stride...")
            all_word_embeddings = self.scaler.transform(all_word_embeddings)

            start = perf_counter()

            self.pca.fit(all_word_embeddings)

            stop = perf_counter()
            diff = stop - start
            print(f"Time needed to fit pca with subsampling of {subsamples}: {str(datetime.timedelta(seconds=diff))}")


            with open(file_loc, "wb") as f:
                pickle.dump(self.pca, f)

    def pca_transform(self, data_set: str):
        """
        Transform new data with existing principal components
        :param data_set: train/devel/test
        :return:
        """
        if self.bert_model != "acoustic":
            with open(f"data/embeddings_pickle/{self.bert_model}_{data_set}_{self.word_or_sent}.pickle", 'rb') as f:
                utterance_arrays = pickle.load(f)

        if self.acoustics != "":
            if self.acoustics == "_wav2vec2":
                with open(f"data/embeddings_pickle/acoustic_{data_set}_{self.word_or_sent}{self.acoustics}.pickle", 'rb') as f:
                    utterance_acoustics = pickle.load(f)
            else:
                with open(f"data/acoustics_pickle/{data_set}{self.acoustics}.pickle", 'rb') as f:
                    utterance_acoustics = pickle.load(f)

            #with open(f"data/acoustics_pickle/{data_set}{self.acoustics}.pickle", 'rb') as f:
            #   utterance_acoustics = pickle.load(f)

            if self.bert_model != "acoustic":
                for i in range(len(utterance_acoustics)):
                    assert len(utterance_arrays[i]) == len(utterance_acoustics[i])
                utterance_arrays = [np.concatenate((utterance_arrays[x], utterance_acoustics[x]), axis=1) for x in range(len(utterance_arrays))]
            else:
                utterance_arrays = utterance_acoustics

        utterance_arrays = utterance_arrays
        # if utterance_arrays[0].shape[1] < self.nr_of_pcs:
        #     print("Warning: your number of pca eigenvectors is larger than the number of features in your data.")
        # utterance_arrays = self.tensor_to_numpy(utterance_tensors)
        pca_utterance_embeddings = self.transform_utterance(utterance_arrays)

        if self.bert_model == "bow" or self.bert_model == "tfidf":
            pca_utterance_embeddings = pca_utterance_embeddings[:, : self.nr_of_pcs]
        else:
            pca_utterance_embeddings = [x[:, : self.nr_of_pcs] for x in pca_utterance_embeddings]

        self.explained_variance = sum(self.pca.explained_variance_ratio_[: self.nr_of_pcs])

        # print(sum(self.pca.explained_variance_ratio_[: 200]))
        # print(sum(self.pca.explained_variance_ratio_[: 250]))
        # print(sum(self.pca.explained_variance_ratio_[: 300]))
        # print(sum(self.pca.explained_variance_ratio_[: 350]))



        with open(f"data/embeddings_pickle/{self.bert_model}_{data_set}_{self.word_or_sent}{self.acoustics}_{self.nr_of_pcs}pca"
                  f".pickle", "wb") as f:
            pickle.dump(pca_utterance_embeddings, f)
        # print("Reduced new data")

    # @staticmethod
    # def tensor_to_numpy(tensors: list) -> list:
    #     """
    #     Convert 2d tensor to 2d numpy array.
    #     :param tensors: tensor representation of embeddings
    #     :return: numpy representation of embeddings
    #     """
    #     utterance_arrays = []
    #     for tensor in tensors:
    #         tensor_to_np = tensor.detach().cpu().numpy()
    #         utterance_arrays.append(tensor_to_np)
    #     return utterance_arrays

    def transform_utterance(self, utterance_arrays: list) -> list:
        """
        Transform all utterances with the obtained principal components.
        :param utterance_arrays: 2d array representation of the embeddings in the utterance
        :return: reduced embeddings
        """
        pca_utterance_embeddings = []
        if self.bert_model == "bow" or self.bert_model == "tfidf":
            utterance_scaled = self.scaler.transform(utterance_arrays)
            utterance_reduced = self.pca.transform(utterance_scaled)
            return utterance_reduced

        for utterance in utterance_arrays:
            utterance_scaled = self.scaler.transform(utterance)
            utterance_reduced = self.pca.transform(utterance_scaled)
            pca_utterance_embeddings.append(utterance_reduced)

        return pca_utterance_embeddings


if __name__ == "__main__":
    os.chdir("..")
    pca = EmbeddingPCA(bert_model="bert", words_or_sentences="words", pca_components=400, overwrite_file=False)
    pca.pca_fit()
    pca.pca_transform(data_set="train")
    pca.pca_transform(data_set="devel")
    pca.pca_transform(data_set="test")
