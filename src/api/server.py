import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.options as options
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLUHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def post(self):
        from src.nlp.core import Flash
        rasa_path = "rasa_nlu/"
        data = tornado.escape.json_decode(self.request.body)

        txt = ''
        data_file = 'demo-rasa'
        to_train = False

        if 'text' in data:
            txt = str(data['text'])
        if 'data_file' in data:
            data_file = str(data['data_file'])
        if 'train' in data:
            to_train = bool(data['train'])

        logging.info("Text: " + txt + " | Data File: " + data_file + " | Train: " + str(to_train))

        if to_train:
            flash_map[data_file] = Flash(rasa_path=rasa_path, train=to_train, data_file=data_file)

        if txt == '' and not to_train:
            self.finish("Please enter text.")
        else:
            resp = True
            if not txt:
                txt = "some random text"
                resp = False
            nlu_obj = flash_map[data_file]
            res, prediction, entities, confidence = nlu_obj.flash_nlu(txt=txt)
            response = {
                'prediction': prediction,
                'confidence': confidence,
                'entities': entities,
                'full_response': res
            }
            if resp:
                self.write(response)
            else:
                if to_train:
                    self.finish("Training done.")


class KeywordHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        txt = str(data['text'])
        if not txt:
            response = {
                'error': True,
                'msg': 'Please enter text.'
            }
        else:
            from src.nlp.core import nl_extractor
            nouns, clean_text, norm_text, important_phrases, sentiment = nl_extractor(txt)
            response = {
                'nouns': nouns,
                'important_phrases': important_phrases,
                'sentiment': sentiment,
                'clean_text': clean_text,
                'norm_text': norm_text
            }
        self.write(response)


# Data loading API
class DataHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def post(self):
        rasa_path = "rasa_nlu/"

        data_folder_path = rasa_path + "data/cyborg/"
        data_file = self.request.files['file'][0]
        original_fname = data_file['filename']

        output_file = open(data_folder_path + original_fname, 'wb')
        output_file.write(data_file['body'])

        self.finish("Data file " + original_fname + " is uploaded.")

flash_map = {}


def main():
    rasa_path = "rasa_nlu/"
    data_folder_path = rasa_path + "data/cyborg/"
    import os
    from src.nlp.core import Flash
    for d in os.listdir(data_folder_path):
        nlu_model = None
        try:
            nlu_model = Flash(rasa_path=rasa_path, data_file=str.split(d, '.')[0])
        except Exception:
            logging.error("Could not find model for: " + d)
        if nlu_model:
            flash_map[str.split(d, '.')[0]] = nlu_model

    options.parse_command_line()
    application = tornado.web.Application([
        (r"/ner", NLUHandler),
        (r"/keyword", KeywordHandler),
        (r"/upload", DataHandler)
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
