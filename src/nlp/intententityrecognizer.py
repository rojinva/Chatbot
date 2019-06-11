from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import os
from src.utils.modeldb import load_models, update_modeldb
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentEntityRecognizer:

    def __init__(self, rasa_path, training_data, config_file=None, train=True, **kwargs):
        logging.info("Initiated Class")
        self.rasa_path = rasa_path
        if config_file is not None:
            self.configuration = config_file
        else:
            self.configuration = None
        self.configs = {}
        self.training_data_filename = training_data
        self.training_data = rasa_path + 'data/cyborg/' + self.training_data_filename + '.json'

        self.model_directories = {}
        # self.models_to_train = ['SPACY', 'MITIE']
        self.models_to_train = ['MITIE']

        if not train:
            models = load_models()[training_data]
            try:
                self.model_directories['MITIE'] = models['MITIE']
            except KeyError:
                pass
            try:
                self.model_directories['SPACY'] = models['SPACY']
            except KeyError:
                pass

        for key in kwargs:
            if key is "mitie_file":
                self.mitie_file = kwargs[key]
            if key is "models_to_train":
                self.models_to_train = kwargs[key]
        if train:
            self.__model = self.train()
        self.__interpreters = self.interpreter()

    def config_generator(self, **kwargs):
        if self.configuration is not None:
            config = RasaNLUConfig(self.configuration)
        else:
            config = RasaNLUConfig()
        for key in kwargs:
            config[str(key)] = kwargs[key]
        return config

    def mitie_model(self):
        config = self.config_generator(mitie_file=self.mitie_file,
                                       pipeline=["nlp_mitie", "tokenizer_mitie", "ner_mitie", "ner_synonyms", "intent_featurizer_mitie", "intent_classifier_sklearn"],
                                       num_threads=8,
                                       data=self.training_data)
        return config

    def spacy_model(self):
        config = self.config_generator(pipeline=["nlp_spacy", "tokenizer_spacy", "intent_featurizer_spacy", "ner_crf", "ner_synonyms", "intent_classifier_sklearn"],
                                       num_threads=4,
                                       data=self.training_data)
        return config

    def train(self):
        training_data = load_data(self.training_data)
        logging.info("Training for: " + ", ".join(self.models_to_train))
        for m in self.models_to_train:
            if m.lower() == 'spacy':
                self.configs['SPACY'] = self.spacy_model()
            elif m.lower() == 'mitie':
                self.configs['MITIE'] = self.mitie_model()
        trainers = {}
        for key in self.configs:
            trainer = Trainer(config=self.configs[key])
            trainer.train(training_data)
            self.model_directories[key] = trainer.persist(self.rasa_path + 'models')
            trainers[key] = trainer
        update_modeldb(self.training_data_filename, self.model_directories)
        return trainers

    @property
    def model(self):
        return self.__model

    def interpreter(self):
        interpreters = {}
        for m in self.models_to_train:
            if m.lower() == 'spacy':
                self.configs['SPACY'] = self.spacy_model()
            elif m.lower() == 'mitie':
                self.configs['MITIE'] = self.mitie_model()
        # self.configs = {"SPACY": self.spacy_model(), "MITIE": self.mitie_model()}
        for model in self.model_directories:
            metadata = Metadata.load(self.model_directories[model])
            interpreters[model] = Interpreter.load(metadata, self.configs[model])
        return interpreters

    def predict(self, txt):
        res = {}
        for model in self.__interpreters:
            res[model] = self.__interpreters[model].parse(unicode(txt))
        return res


if __name__ == '__main__':
    rasa_path = "/Users/z002nt3/Work/Target/chatbot/rasa_nlu/"

    ner = IntentEntityRecognizer(rasa_path=rasa_path,
                                 training_data='hr_policies',
                                 mitie_file="rasa_nlu/data/total_word_feature_extractor.dat", train=True) #, models_to_train=['SPACY'])

    ner.predict(txt="i am looking for an indian spot called Briyani Zone")
    ner.predict(txt="i am looking for an indian spot called Briyani Zone")

    ner.predict(txt="i am loo Zone")


    def config_generator(**kwargs):
        from rasa_nlu.converters import load_data
        from rasa_nlu.config import RasaNLUConfig
        from rasa_nlu.model import Trainer
        from rasa_nlu.model import Metadata, Interpreter
        config = RasaNLUConfig()
        for key in kwargs:
            config[str(key)] = kwargs[key]
        return config

    def mitie_model():
        config = config_generator(mitie_file="rasa_nlu/data/total_word_feature_extractor.dat",
                                  pipeline=["nlp_mitie", "tokenizer_mitie", "ner_mitie", "ner_synonyms", "intent_featurizer_mitie", "intent_classifier_sklearn"],
                                  num_threads=8,
                                  data='/Users/z002nt3/Work/Target/chatbot/rasa_nlu/data/cyborg/hr_policies.json')
        return config

    from time import time
    from timeit import timeit

    c = mitie_model()

    # ner.predict(txt="Hi! How are you doin?")
    # ner.predict(txt="Thanks for the time")
    # res = ner.predict(txt="i am looking for an indian spot called Briyani Zone")
    #
    #
    # def dict_compare(d1, d2):
    #     d1_keys = set(d1.keys())
    #     d2_keys = set(d2.keys())
    #     intersect_keys = d1_keys.intersection(d2_keys)
    #     added = d1_keys - d2_keys
    #     removed = d2_keys - d1_keys
    #     modified = {o : (d1[o] + d2[o])/2.0 for o in intersect_keys if d1[o] != d2[o]}
    #     same = set(o for o in intersect_keys if d1[o] == d2[o])
    #     return added, removed, modified, same
    #
    #
    # ir1 = {}
    # for v in res["MITIE"]["intent_ranking"]:
    #     ir1[v['name']] = v['confidence']
    #
    # ir2 = {}
    # for v in res["SPACY"]["intent_ranking"]:
    #     ir2[v['name']] = v['confidence']
    #
    # added, removed, modified, same = dict_compare(ir1, ir2)
