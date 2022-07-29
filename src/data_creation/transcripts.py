import json
import os
import pandas as pd
import statistics
import itertools
from typing import List
import numpy as np


class Transcript:
    def __init__(self, directory: str, datasets: list) -> None:
        self.directory = directory
        self.datasets = datasets

    def transcripts_to_df(self, ds: str, word_time_stamps=False) -> pd.DataFrame:
        """
        Create single dataframe from multiple json transcript files.
        :param ds: train/devel/test set
        :param word_time_stamps: bool for extracting word timestamps
        :return: dataframe of utterances
        """
        all_utterances = []
        all_confidence_scores = []
        utterance_lengths = []
        all_timestamps = []

        ds_folder = f"{self.directory}{ds}"
        for file in os.listdir(ds_folder):
            if file == ".gitignore":
                continue
            filename = open(os.path.join(ds_folder, file))
            file_data = json.load(filename)

            # Insert dummy data for empty files (Only applies to train instance 1991)
            if not file_data:
                file_data = dict(results=[dict(transcript="None", confidence=0,
                                               words=[dict(word="None", start_time=0, end_time=0.1)])])

            if word_time_stamps:
                times = self.word_timestamps(file_data)
                all_timestamps.append([times])
            else:
                full_utterance = self.utterance_string(file_data)
                all_utterances.append(full_utterance)

            utterance_confidence = self.utterance_confidence(file_data)
            all_confidence_scores.append(utterance_confidence)

            full_utterance_words = self.calculate_length(file_data)
            utterance_lengths.append(full_utterance_words)

        confidence_scores = np.array(all_confidence_scores)
        self.print_statistics(ds, confidence_scores, utterance_lengths)
        np.savetxt(f"data/features_csv/confidence_metrics_{ds}.csv", confidence_scores, delimiter=",")

        if word_time_stamps:
            dataframe = pd.DataFrame(all_timestamps, columns=["timestamps"])
        else:
            dataframe = pd.DataFrame(all_utterances, columns=["sentence"])
        return dataframe

    @staticmethod
    def utterance_string(data: dict) -> str:
        """
        Extract utterance from dictionary.
        :param data: dict of json contents
        :return: concatenated string of all utterances
        """
        partial_utterances = [x["transcript"] for x in data["results"]]
        full_utterance = ' '.join(partial_utterances)
        return full_utterance

    @staticmethod
    def word_timestamps(data: dict) -> list:
        partial_utterance_words = [x["words"] for x in data["results"]]
        full_utterance_words = list(itertools.chain.from_iterable(partial_utterance_words))
        time_tuples = []
        for word in full_utterance_words:
            time_tuples.append((word["start_time"], word["end_time"]))
        return time_tuples

    @staticmethod
    def utterance_confidence(data: dict) -> list:
        """
        Average confidence of the full utterance.
        :param data: dict of json contents
        :return: overall utterance confidence
        """
        partial_confidences = [round(x["confidence"], 4) for x in data["results"]]
        mean_c = round(statistics.mean(partial_confidences), 4)
        max_c = max(partial_confidences)
        min_c = min(partial_confidences)
        return [mean_c, max_c, min_c]

    @staticmethod
    def calculate_length(data: dict) -> int:
        """
        Length of the full utterance.
        :param data: dict of json contents
        :return: length of the utterance
        """
        partial_utterance_words = [x["words"] for x in data["results"]]
        full_utterance_words = list(itertools.chain.from_iterable(partial_utterance_words))
        utterance_length = len(full_utterance_words)
        return utterance_length

    @staticmethod
    def print_statistics(ds: str, utterance_confidences: np.array, utterance_lengths: List[int]):
        """
        Print some statistics of the dataset.
        :param ds: train/devel/test
        :param utterance_confidences: confidence scores of the utterances
        :param utterance_lengths: lengths of the utterances
        :return:
        """
        print(f"Statistics for the {ds} set:")
        print(f"confidence:       {utterance_confidences[:20]}")

        print(f"Mean utterance length: {round(statistics.mean(utterance_lengths), 4)}")
        print(f"SD utterance length:   {round(statistics.pstdev(utterance_lengths), 4)}")
        print()


def sentences():
    transcript = Transcript(directory="data/transcripts_json/", datasets=["train", "devel", "test"])
    for dataset in transcript.datasets:
        df = transcript.transcripts_to_df(dataset)
        df.to_csv(f"data/features_csv/{dataset}_sentences.csv", index=False)


def timestamps():
    transcript = Transcript(directory="data/transcripts_json/", datasets=["train", "devel", "test"])
    for dataset in transcript.datasets:
        df = transcript.transcripts_to_df(dataset, word_time_stamps=True)
        df.to_csv(f"data/features_csv/{dataset}_timestamps.csv", index=False)


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    # sentences()
    timestamps()
