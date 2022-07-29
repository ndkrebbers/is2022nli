import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from os import path
from time import perf_counter
import datetime
import os
from sklearn.cluster import KMeans


class FisherVector:
    def __init__(self, bert_model: str, data_extension: str, gmm_components: int, acoustics: str = "", vlad=False):
        with open(f"data/embeddings_pickle/{bert_model}_train_words{acoustics}_{data_extension}.pickle", "rb") as f:
            self.train_data = pickle.load(f)
        self.gmm_components = gmm_components
        self.gmm = GaussianMixture(n_components=gmm_components, covariance_type='diag', verbose=1, random_state=42, max_iter=300)
        self.bert_model = bert_model
        self.extension = data_extension
        self.acoustics = acoustics
        self.vlad = vlad
        self.size = 0

    def fit_gmm(self, subsamples=1, overwrite=False):
        """
        Fit the gmm to the training data, load pickle if gmm is already made
        :return:
        """
        gmm_loc = f"data/process_pickle/gmm_{self.bert_model}_train_words{self.acoustics}_{self.extension}_{self.gmm_components}gmm.pickle"
        if path.exists(gmm_loc) and not overwrite:
            with open(gmm_loc, "rb") as f:
                print(f"Using existing gmm pickle file for {self.bert_model}_train_words{self.acoustics}"
                      f"_{self.extension}_{self.gmm_components}gmm.")
                self.gmm = pickle.load(f)
        else:
            print(f"Fitting new gmm pickle to {self.bert_model}_train_words{self.acoustics}_{self.extension}_{self.gmm_components}gmm...")
            utterance_arrays = []
            for utt in self.train_data:
                utterance_arrays.append(utt)
            all_word_embeddings = np.concatenate(utterance_arrays, axis=0)

            if self.acoustics == "_acoustic_llds" or self.acoustics == "_compare_llds" or self.acoustics == "_llds" or self.acoustics == "_wav2vec2":
                all_word_embeddings = all_word_embeddings[0::subsamples, ]

            start = perf_counter()

            self.gmm.fit(all_word_embeddings)

            stop = perf_counter()
            diff = stop - start
            print(f"Time needed to fit gmm with subsampling of {subsamples}: {str(datetime.timedelta(seconds=diff))}")

            self.pickle_gmm()

    def pickle_gmm(self):
        """
        Pickle gmm fitted on training data, to use for FV encoding of new data
        :return:
        """
        with open(f"data/process_pickle/gmm_{self.bert_model}_train_words{self.acoustics}_{self.extension}_"
                  f"{self.gmm_components}gmm.pickle", "wb") as f:
            pickle.dump(self.gmm, f)

    def fit_kmeans(self):
        all_utterance_embeddings = []
        for utt in self.train_data:
            all_utterance_embeddings.append(utt)
        all_utterance_embeddings = np.concatenate(all_utterance_embeddings, axis=0)

        # Subsampling
        all_utterance_embeddings = all_utterance_embeddings[0::10]

        n_clusters = 1000
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1).fit(all_utterance_embeddings)

        for data in ["train", "devel", "test"]:
            with open(f"data/embeddings_pickle/{self.bert_model}_{data}_words{self.acoustics}_{self.extension}.pickle", "rb") as f:
                embeddings = pickle.load(f)

            count_list = []
            for utt in embeddings:
                preds = kmeans.predict(utt)
                counts = np.bincount(preds, minlength=n_clusters)
                count_list.append(counts)
            q_sum = np.array(count_list)

            with open(f"data/embeddings_pickle/bert_{data}_bow.pickle", "wb") as f:
                pickle.dump(q_sum, f)
        print("hello")

    def compute_fv(self):
        """
        Convert arbitrary length utterance representation to fixed length fisher vector.
        :return:
        """
        for data in ["train", "devel", "test"]:
            with open(f"data/embeddings_pickle/{self.bert_model}_{data}_words{self.acoustics}_{self.extension}.pickle", "rb") as f:
                embeddings = pickle.load(f)

            fv_encodings = []
            for embed in embeddings:
                # embed = np.float32(embed)
                fv = self.fisher_vector(embed)
                fv_encodings.append(fv)

            fv_encodings = np.array(fv_encodings)

            self.size = fv_encodings.shape[1]
            
            if self.vlad:	
                with open(f"data/embeddings_pickle/{self.bert_model}_{data}_words{self.acoustics}_{self.extension}_"
            	f"{self.gmm_components}gmm_vlad.pickle", "wb") as f:
                    pickle.dump(fv_encodings, f)
            else:
                with open(f"data/embeddings_pickle/{self.bert_model}_{data}_words{self.acoustics}_{self.extension}_"
                f"{self.gmm_components}gmm_fv.pickle", "wb") as f:
                    pickle.dump(fv_encodings, f)

    def fisher_vector(self, xx):
        """
        Calculate Fisher Vector using gmm as background model
        :param xx: 2d array of data to encode
        :return: vector of length 2*D*K
        Reference: https://gist.github.com/danoneata/9927923
        """
        xx = np.atleast_2d(xx)
        N = xx.shape[0]

        # Compute posterior probabilities.
        Q = self.gmm.predict_proba(xx)  # NxK

        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_xx = np.dot(Q.T, xx) / N
        Q_xx_2 = np.dot(Q.T, xx ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        # d_pi = Q_sum.squeeze() - self.gmm.weights_ # derivative of pi is not necessary
        d_mu = Q_xx - Q_sum * self.gmm.means_
        d_sigma = (
            - Q_xx_2
            - Q_sum * self.gmm.means_ ** 2
            + Q_sum * self.gmm.covariances_
            + 2 * Q_xx * self.gmm.means_)

        # Merge derivatives into a vector.
        
        if self.vlad:
            arr = np.hstack(d_mu.flatten())
            # TODO set name to VLAD instead of FV in filename
        else:
            arr = np.hstack((d_mu.flatten(), d_sigma.flatten())) # TODO Change this to switch between VLAD and FV

#        fv = arr.flatten(order="F")  # Use this order to create 112233 type of vector instead of 123123, to group gmm components

        return arr


def component_indices(pca_comp, gmm_comp):
    tmp_arr = np.zeros((1, gmm_comp * pca_comp))
    for i in range(gmm_comp):
        start_ind = i * pca_comp
        end_ind = (i + 1) * pca_comp
        tmp_arr[0, start_ind:end_ind] = [i] * (end_ind - start_ind)

    arr2 = np.array((tmp_arr.flatten(), tmp_arr.flatten()))
    order = arr2.flatten(order="F")

    return order

if __name__ == '__main__':
    os.chdir("..")
    fish_v = FisherVector(bert_model="acoustic", data_extension="wav2vec2_400pca", gmm_components=128, vlad=True)
    # fish_v.fit_kmeans()
    fish_v.fit_gmm(overwrite=False, subsamples=1)
    fish_v.compute_fv()
    # fff = fish_v.return_order([1, 1, 2, 2, 3, 3])
    # print(fff)


    # fish_v.fit_gmm()
    # fish_v.compute_fv()
