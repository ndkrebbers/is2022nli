from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import pickle
import sys


class SentenceBert:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def obtain_embeddings(self):
        for dataset in ["train", "devel", "test"]:
            paragraphs = self.__load_data(dataset)
            # embeddings = self.model.encode(paragraphs)
            embedding_list = self.__separated_sentence_embeddings(paragraphs)
            self.__pickle_embeddings(embedding_list, dataset)

    def __separated_sentence_embeddings(self, paragraphs):
        embedding_list = []
        i = 0
        for par in paragraphs:
            sentences = par.split(".")
            embeddings = self.model.encode(sentences)
            embedding_list.append(embeddings)

            if i % 50 == 0:
                self.__print_progress(i, len(paragraphs))
            i += 1
        self.__print_progress(i, len(paragraphs))
        return embedding_list

    @staticmethod
    def __print_progress(i, max_i):
        sys.stdout.flush()
        sys.stdout.write(f"\rProgress: Read %i/{max_i} paragraphs." % i)

    @staticmethod
    def __load_data(dataset):
        data_loc = f"data/features_csv/{dataset}_sentences.csv"
        df = pd.read_csv(data_loc)
        sentences = df.sentence.values
        return sentences

    @staticmethod
    def __pickle_embeddings(embeddings, dataset):
        file_loc = f"data/embeddings_pickle/sentencebert_{dataset}.pickle"
        with open(file_loc, "wb") as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    sb = SentenceBert()
    sb.obtain_embeddings()
