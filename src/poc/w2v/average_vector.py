import numpy as np
from scipy import spatial
from src.nlp.core import nl_extractor


def avg_feature_vector(words, model, num_features):
    # function to average all words vectors in a given paragraph
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # list containing names of words in the vocabulary
    index2word_set = set(model.index2word)  # this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])

    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def sentence_similarity(sentence_1, sentence_2, word2vec_model):
    sentence_1_avg_vector = avg_feature_vector(words=sentence_1.split(), model=word2vec_model, num_features=300)
    sentence_2_avg_vector = avg_feature_vector(words=sentence_2.split(), model=word2vec_model, num_features=300)
    similarity = 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)
    return similarity


def word2vec_model():
    import gensim
    w2v_model_file = '/Users/z002nt3/Work/Target/chatbot-kb/softwares/model/GoogleNews-vectors-negative300.bin'
    return gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file, binary=True)


if __name__ == '__main__':
    from src.nlp.core import nl_extractor

    model = word2vec_model()

    sentence1 = "How is tax calculated"
    sentence2 = "how can I calculate my tax"

    sentence_similarity("".join(nl_extractor(sentence1)[2]), "".join(nl_extractor(sentence2)[2]), model)

