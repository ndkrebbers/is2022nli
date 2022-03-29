from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk import tokenize
import numpy as np
import pickle
import itertools


class SBERT:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pass

    def split_utterances(self, dataset: str):
        """
        Read csv and calculate average sentence embeddings.
        :param dataset: train/devel/test
        :return:
        """
        df = pd.read_csv(f"data/features_csv/{dataset}_sentences.csv")
        data = df.sentence.values

        sentences = [tokenize.sent_tokenize(x) for x in data]
        concat_sentences = list(itertools.chain.from_iterable(sentences))

        indices_and_lengths = self.indices_and_lengths(sentences)
        embeddings = self.model.encode(concat_sentences)

        avg_embeddings = self.concat_to_average_embeddings(embeddings, indices_and_lengths)

        self.save_to_file(avg_embeddings, dataset)

    @staticmethod
    def concat_to_average_embeddings(all_embeddings, i_and_l) -> np.array:
        """
        Convert concatenated sentence list to average sentence embeddings per utterance.
        :param all_embeddings: list of all sentence embeddings in the dataset
        :param i_and_l: indices and lengths of utterances
        :return: average sentence embedding
        """
        sentence_embeddings = [all_embeddings[x:x + y] for (x, y) in i_and_l]
        avg_embeddings = [np.mean(x, axis=0) for x in sentence_embeddings]
        avg_embeddings = np.array(avg_embeddings)
        return avg_embeddings

    @staticmethod
    def indices_and_lengths(sentences: list) -> list:
        """
        Return the starting index and number of sentences of each utterance in the concatenated sentence list.
        :param sentences: all sentences in the dataset
        :return: (starting index, number of sentences in utterance)
        """
        indices_and_lengths = []
        index = 0
        for i in sentences:
            indices_and_lengths.append((index, len(i)))
            index += len(i)
        return indices_and_lengths

    @staticmethod
    def save_to_file(embeddings: np.array, dataset: str):
        """
        Write array to file.
        :param embeddings: average sentence embeddings for utterance representation.
        :param dataset: train/devel/test
        :return:
        """
        with open(f"data/embeddings_pickle/sbert_{dataset}_sentences.pickle", "wb") as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    bert = SBERT()
    bert.split_utterances("train")
    bert.split_utterances("devel")
