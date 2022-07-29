from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os
import pandas as pd
import numpy as np


class Vectorizer:
    def __init__(self, vec_type: str, ngram: int):
        self.current_dataset = None
        self.vec_type = vec_type
        self.ngram = ngram
        pass

    def all_data_to_vec(self):
        combined_data = []
        splits = ["train", "devel", "test"]
        for dataset in splits:
            self.current_dataset = dataset
            data = self.__read_transcript_csv()
            combined_data.append(data)
        combined_data = np.concatenate(combined_data)

        if self.vec_type == "bow":
            vec_combined = self.__sentences_to_bow(combined_data)
        else:  # self.vec_type == "tfidf
            vec_combined = self.__sentences_to_tfidf(combined_data)
        end_train = 3300
        end_devel = end_train + 965
        separated_sets = [vec_combined[:end_train, ], vec_combined[end_train:end_devel, ], vec_combined[end_devel:, ]]

        for i in range(len(splits)):
            self.current_dataset = splits[i]
            self.__df_to_pickle(separated_sets[i])

    def __read_transcript_csv(self):
        data_loc = f"data/features_csv/{self.current_dataset}_sentences.csv"
        df = pd.read_csv(data_loc)
        sentences = df.sentence.values
        return sentences

    def __sentences_to_bow(self, data):
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(self.ngram, self.ngram))
        X = vectorizer.fit_transform(data)
        bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        return bow_df.values

    def __sentences_to_tfidf(self, data):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(self.ngram, self.ngram))
        X = vectorizer.fit_transform(data)
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        return tfidf_df.values

    def __df_to_pickle(self, bow):
        if self.ngram == 1:
            gram = "unigram"
        else:
            gram = "bigram"

        with open(f"data/embeddings_pickle/{self.vec_type}_{self.current_dataset}_{gram}.pickle", "wb") as f:
            pickle.dump(bow, f)


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    bag_of_words = Vectorizer(vec_type="tfidf", ngram=2)
    bag_of_words.all_data_to_vec()
