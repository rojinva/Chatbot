from flask import request, abort
from flask.ext.api import FlaskAPI

from src.nlp.intententityrecognizer import IntentEntityRecognizer
from src.nlp.sentiment_extractor import sentiment_extractor as sentiment

app = FlaskAPI(__name__)
import logging
import json


@app.route('/nlu', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        query = str(request.data.get('text', ''))
        RASA_PATH = "/Users/z002nt3/Work/Softwares/rasa_nlu/"
        res = {}
        try:
            ner = IntentEntityRecognizer(rasa_path=RASA_PATH, config_file=RASA_PATH + "config_spacy.json",
                                         training_data=RASA_PATH + 'data/examples/rasa/demo-rasa.json')

            res['ner'] = ner.predict.parse(unicode(query))
            logging.warn("NER Done for: " + query)

            res['sentiment'] = vars(sentiment(query)[0])
            logging.warn("Sentiment Done")

            return json.dumps(res)
        except Exception as e:
            abort(404)
            raise e

if __name__ == "__main__":
    app.run(debug=True, threaded=True, host='0.0.0.0', port=8081)
