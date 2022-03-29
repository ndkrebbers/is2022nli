import transcripts
# from src import extract_embeddings
from src import pca_for_embeddings
from src import model_learning
from os import path

from sklearn.metrics import recall_score

# Adjust


def main(bert_model: str, dataset: str, words_or_sentences: str, pca_components: int, fisher_vector: bool):
    get_transcripts(dataset, "data/transcripts_json/", ["train", "devel", "test"])

    assert pca_components >= 0, "Number of PCA components must be larger than 0, or zero if PCA should not be applied."
    if pca_components > 0:
        reduce_embeddings(bert_model, dataset, words_or_sentences, pca_components)

    if fisher_vector:
        # TODO FV method
        pass

    ml = model_learning.ModelLearning(bert_model=bert_model, test=dataset, data_extension="sentences")
    ml.normalize_data()
    pred = ml.fit_predict()

    print(recall_score(ml.y_test, pred, average="macro"))


def get_transcripts(dataset, directory: str, datasets: list):
    transcripts_loc = f"data/features_csv/{dataset}_sentences.csv"
    if not path.exists(transcripts_loc):
        transcript = transcripts.Transcript(directory=directory, datasets=datasets)
        for ds in transcript.datasets:
            df = transcript.transcripts_to_df(ds)
            df.to_csv(f"data/features_csv/{ds}_sentences.csv", index=False)


def reduce_embeddings(bert_model: str, data_set: str, words_or_sentences: str, pca_components: int):
    features_loc = f"data/embeddings_pickle/{bert_model}_{data_set}_{words_or_sentences}_{pca_components}pca.pickle"
    if not path.exists(features_loc):
        pca = pca_for_embeddings.EmbeddingPCA(bert_model=bert_model, words_or_sentences=words_or_sentences,
                                              pca_components=pca_components, overwrite_file=False)
        pca.pca_fit()
        pca.pca_transform(data_set=data_set)


if __name__ == "__main__":
    main(bert_model="bert",
         dataset="train",
         words_or_sentences="words",
         pca_components=0,
         fisher_vector=False
         )
