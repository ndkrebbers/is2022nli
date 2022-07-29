import os
import matplotlib.pyplot as plt
import librosa
import librosa.display


def visualize_mfcc(audio_path):
    y, sr = librosa.load(audio_path, duration=5.0, sr=16000)
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 4)
    fig.set_dpi(100)
    img = librosa.display.specshow(mfcc_data, x_axis='time', sr=16000, cmap="magma", win_length=400)
    fig.colorbar(img)

    ax.set_title('Mel-fequency cepstrum coefficients')
    plt.ylabel("MFCC")
    plt.yticks(range(0, 24, 2))
    plt.xlim([0, 5])
    plt.show()


if __name__ == "__main__":
    os.chdir("..")
    # 1200 = French, 2200 = Arabic
    audio_source_path = "data/audio_wav/train/train_1200.wav"
    visualize_mfcc(audio_source_path)
