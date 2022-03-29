import pandas as pd
import ast
import os
import numpy as np
import pickle
import sys


class WordAlignedAcoustics:
    def __init__(self, directory: str, data: str):
        self.directory = directory
        self.data = data
        self.t_stamps = self.read_stamps()

    def calculate_word_functionals(self):
        ds_folder = f"{self.directory}{self.data}"
        file_index = 0
        all_means = []
        all_means_sd = []
        all_means_r = []
        files_ordered = []

        for file in os.listdir(ds_folder):
            if file == ".gitignore":
                continue
            files_ordered.append(file)
        files_ordered.sort()

        for file in files_ordered:
            df = pd.read_csv(os.path.join(ds_folder, file), header=None)

            mean, mean_sd, mean_r = self.calculate_means(df, file_index)
            all_means.append(mean)
            all_means_sd.append(mean_sd)
            all_means_r.append(mean_r)

            file_index += 1
            if file_index % 50 == 0:
                sys.stdout.write(f"\rProgress: extracted %i/{len(self.t_stamps)} word-level functionals." % file_index)
                sys.stdout.flush()
        sys.stdout.flush()
        sys.stdout.write(f"Progress: extracted {len(self.t_stamps)}/{len(self.t_stamps)} word-level functionals.")

        self.to_pickle(all_means, all_means_sd, all_means_r)

    def to_pickle(self, all_means, all_means_sd, all_means_r):
        with open(f"data/acoustics_pickle/{self.data}_means.pickle", "wb") as f:
            pickle.dump(all_means, f)
        with open(f"data/acoustics_pickle/{self.data}_means_sd.pickle", "wb") as f:
            pickle.dump(all_means_sd, f)
        with open(f"data/acoustics_pickle/{self.data}_means_r.pickle", "wb") as f:
            pickle.dump(all_means_r, f)

    def calculate_means(self, df, file_index):
        means = []
        means_sd = []
        means_range = []
        for start, end in self.t_stamps[file_index]:
            # assert start < end, f"{start}, {end}, {file_index}"
            start_frame = int(start * 100)
            end_frame = int(end * 100) + 2
            word_frame_values = df.iloc[start_frame:end_frame, ].values

            # Simple catch for out of range timestamps
            if word_frame_values.shape[0] == 0:
                word_frame_values = np.zeros((1, word_frame_values.shape[1]))
            # assert word_frame_values.shape[0] > 0, f"{start},{end}, {file_index}"

            word_mean = np.mean(word_frame_values, axis=0)

            word_sd = np.std(word_frame_values, axis=0)
            word_range = np.ptp(word_frame_values, axis=0)

            means.append(word_mean)
            means_sd.append(np.concatenate((word_mean, word_sd)))
            means_range.append(np.concatenate((word_mean, word_range)))
        return np.array(means), np.array(means_sd), np.array(means_range)

    def read_stamps(self):
        df = pd.read_csv(f"data/features_csv/{self.data}_timestamps.csv")
        timestamp_strings = df.timestamps.values
        timestamps = []
        for i in timestamp_strings:
            x = ast.literal_eval(i)
            timestamps.append(x)
        return timestamps


if __name__ == "__main__":
    for ds in ["train", "devel", "test"]:
        wa = WordAlignedAcoustics("data/acoustic_lld_csv/", ds)
        wa.calculate_word_functionals()

    print("okay")
