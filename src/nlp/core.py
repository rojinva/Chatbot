import logging
import operator

from pycorenlp import StanfordCoreNLP
from sentiment_extractor import sentiment_extractor
from noun_extractor import noun_extractor
from imp_phrase_extractor import important_phrases

from src.nlp.intententityrecognizer import IntentEntityRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nl_extractor(txt, corenlp_server='http://10.63.108.116:9000/'):

    from ds_utils.parse import normalize
    text = normalize(txt.lower())

    nlp = StanfordCoreNLP(corenlp_server)

    # Extracting nouns
    nouns = noun_extractor(text, nlp)

    # Sentiment analysis
    sentiment = sentiment_extractor(text, nlp)

    # Clean text

    from nltk.corpus import stopwords
    from textblob import TextBlob

    def tokenize(txt, ngram_range=(3, 7)):
        from collections import namedtuple

        tokenize_resp = namedtuple('Token', 'words sentences ngrams')
        tb = TextBlob(txt.lower())
        ngrams_list = []
        for i in xrange(ngram_range[0], ngram_range[1]):
            ngrams_list.append(tb.ngrams(i))
        return tokenize_resp(words=tb.words, sentences=tb.sentences, ngrams=ngrams_list)

    def remove_stopwords(txt):
        txt_words = tokenize(txt).words
        all_stopwords = stopwords.words('english')
        return [x for x in txt_words if x not in all_stopwords]

    clean_text = " ".join(remove_stopwords(text))

    # Key phrases using TextRank algorithm
    key_phrases = list(important_phrases(clean_text))

    return nouns, clean_text, text, key_phrases, sentiment


def nlu(rasa_path, txt, model='MITIE', train=False, data_file='demo-rasa'):

    nouns, clean_text, norm_text, key_phrases, sentiment = nl_extractor(txt)

    ner = IntentEntityRecognizer(rasa_path=rasa_path,
                                 training_data=data_file,
                                 mitie_file=rasa_path + "data/total_word_feature_extractor.dat", train=train)

    res = ner.predict(norm_text)

    def dict_compare(d1, d2):
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        intersect_keys = d1_keys.intersection(d2_keys)
        added = d1_keys - d2_keys
        removed = d2_keys - d1_keys
        # Todo: Need to replace it with weighted sum rather average
        modified = {o: (d1[o] + d2[o]) / 2.0 for o in intersect_keys if d1[o] != d2[o]}
        same = set(o for o in intersect_keys if d1[o] == d2[o])
        return added, removed, modified, same

    if model.lower() == 'mitie':
        logging.info("Fetching result using MITIE model")
        prediction = res['MITIE']['intent']['name']
        confidence = res['MITIE']['intent']['confidence']
    elif model.lower() == 'spacy':
        logging.info("Fetching result using SPACY model")
        prediction = res['SPACY']['intent']['name']
        confidence = res['SPACY']['intent']['confidence']
    else:
        ir1 = {}
        for v in res["MITIE"]["intent_ranking"]:
            ir1[v['name']] = v['confidence']

        ir2 = {}
        for v in res["SPACY"]["intent_ranking"]:
            ir2[v['name']] = v['confidence']

        added, removed, modified, same = dict_compare(ir1, ir2)
        logging.info("Fetching result using average model")
        prediction = max(modified.iteritems(), key=operator.itemgetter(1))[0]
        confidence = max(modified.iteritems(), key=operator.itemgetter(1))[1]

    entity = res['MITIE']['entities']

    return res, prediction, entity, nouns, clean_text, norm_text, confidence, key_phrases, sentiment

class Flash:
    def __init__(self, rasa_path, train=False, data_file='demo-rasa'):
        self.ner = IntentEntityRecognizer(rasa_path=rasa_path,
                                          training_data=data_file,
                                          mitie_file=rasa_path + "data/total_word_feature_extractor.dat", train=train)

    def flash_nlu(self, txt, model='MITIE'):

        res = self.ner.predict(txt)

        def dict_compare(d1, d2):
            d1_keys = set(d1.keys())
            d2_keys = set(d2.keys())
            intersect_keys = d1_keys.intersection(d2_keys)
            added = d1_keys - d2_keys
            removed = d2_keys - d1_keys
            # Todo: Need to replace it with weighted sum rather average
            modified = {o: (d1[o] + d2[o]) / 2.0 for o in intersect_keys if d1[o] != d2[o]}
            same = set(o for o in intersect_keys if d1[o] == d2[o])
            return added, removed, modified, same

        if model.lower() == 'mitie':
            logging.info("Fetching result using MITIE model")
            prediction = res['MITIE']['intent']['name']
            confidence = res['MITIE']['intent']['confidence']
        elif model.lower() == 'spacy':
            logging.info("Fetching result using SPACY model")
            prediction = res['SPACY']['intent']['name']
            confidence = res['SPACY']['intent']['confidence']
        else:
            ir1 = {}
            for v in res["MITIE"]["intent_ranking"]:
                ir1[v['name']] = v['confidence']

            ir2 = {}
            for v in res["SPACY"]["intent_ranking"]:
                ir2[v['name']] = v['confidence']

            added, removed, modified, same = dict_compare(ir1, ir2)
            logging.info("Fetching result using average model")
            prediction = max(modified.iteritems(), key=operator.itemgetter(1))[0]
            confidence = max(modified.iteritems(), key=operator.itemgetter(1))[1]

        entity = res['MITIE']['entities']

        return res, prediction, entity, confidence


if __name__ == '__main__':
    rasa_path = "rasa_nlu/"
    txt = "What's the process to apply these leaves - PL, CL, Maternity, Paternity and Bereavement?"
    # txt = "What is Income tax? Why is it deducted from my salary? "
    # txt = "How do I take time off"
    res, prediction, entities, nouns, clean_text, norm_text, confidence, kp, sentiment = nlu(txt=txt, train=True, rasa_path=rasa_path, data_file='demo-rasa')

    print(prediction)
