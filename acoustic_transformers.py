from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import soundfile
import os
import numpy as np
import sys
import pickle


class Wav2Vec2Extractor:
    def __init__(self, directory_loc):
        self.main_dir = directory_loc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)

    def collect_embeddings(self):
        print(f"Using {self.device}")
        for data_dir in os.listdir(self.main_dir):
            #filterfiles= [f for f in os.listdir(os.path.join(self.main_dir, data_dir)) if (os.path.isfile(f) and f.endswith(".wav"))]
            sorted_files = sorted(os.listdir(os.path.join(self.main_dir, data_dir)))
            audio_arrays = []
            index = 0
            all_outputs = []
            for file in sorted_files:
                #if file == ".gitignore":
                if not file.endswith(".wav"):
                    continue
                signal, rate = soundfile.read(os.path.join(self.main_dir, data_dir, file), dtype=np.int16)
                #audio_arrays.append(signal)

                # print()
                output = self.signal_to_arr(signal)
                all_outputs.append(output)

                if index % 50 == 0:
                    self.print_progress(index, len(sorted_files))
                index += 1
            self.print_progress(len(sorted_files), len(sorted_files))

            self.embeddings_to_pickle(all_outputs, data_dir)
            print()

    @staticmethod
    def print_progress(iter_i, max_iter):
        sys.stdout.flush()
        sys.stdout.write(f"\rProgress: Read %i/{max_iter} wav files." % iter_i)

    def signal_to_arr(self, audio_arrays):
        # print("Converting input array to tensor.")
        inputs = self.processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        inputs = inputs.to(torch.float32).to(self.device)
        #  print("Converting input to embeddings.")
        with torch.no_grad():
            outputs = self.model(inputs)
        last_layer = outputs.last_hidden_state

        # print("Taking mean over every 10 embeddings in each utterance.")
        splits = torch.split(last_layer, 10, 1)
        mean_10 = [torch.unsqueeze(torch.mean(x, dim=1), dim=1) for x in splits]
        mean_10_cat = torch.cat(mean_10, dim=1)
        mean_10_cat_list = mean_10_cat.tolist()

        outputs = np.array(mean_10_cat_list[0], dtype=np.float32)
        # outputs = [np.array(x, dtype=np.float32) for x in mean_10_cat_list]

        return outputs

    @staticmethod
    def embeddings_to_pickle(outputs, data_dir):
        print("Writing embeddings to pickle file.")
        with open(f"data/embeddings_pickle/acoustic_{data_dir}_words_wav2vec2.pickle", "wb") as f:
            pickle.dump(outputs, f)
        pass


if __name__ == "__main__":
    directory = "data/audio_wav/"
    w = Wav2Vec2Extractor(directory)
    w.collect_embeddings()
