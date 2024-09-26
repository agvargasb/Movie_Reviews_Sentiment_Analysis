from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    
    corpus_vectors = [model.wv.get_mean_vector(keys=doc, pre_normalize=False) for doc in corpus]

    return np.array(corpus_vectors)


class corpus_iterator:
    def __init__(self, corpus_path):
        self.path = corpus_path

    def __iter__(self):
        with open(self.path) as file:
            for line in file:
                yield line


def vectorizer_iter(corpus_iter, model):
    doc_iter = iter(corpus_iter)
    corpus_vectors = []
    
    try:
        while True:
            doc = next(doc_iter).strip("\n").strip(" ").split(" ")
            doc_vector = model.wv.get_mean_vector(keys=doc, pre_normalize=True, post_normalize=True)
            corpus_vectors.append(doc_vector)
    except StopIteration:
        pass

    return np.array(corpus_vectors)


def pretrained_vectorizer_iter(corpus_iter, vectors):
    doc_iter = iter(corpus_iter)
    corpus_vectors = []
    
    try:
        while True:
            doc = next(doc_iter).strip("\n").strip(" ").split(" ")
            doc_vector = vectors.get_mean_vector(keys=doc, pre_normalize=True, post_normalize=True)
            corpus_vectors.append(doc_vector)
    except StopIteration:
        pass

    return np.array(corpus_vectors)