import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models():
    with open(get_modeldb_file()) as json_file:
        return json.load(json_file)


def write_models(data):
    with open(get_modeldb_file(), 'w') as outfile:
        try:
            json.dump(data, outfile)
        except Exception as e:
            logging.error("Model DB Creation failed. " + e.message)
            raise e


def get_modeldb_file(file_name="resources/modelDB.json"):
    return file_name


def create_modeldb():
    write_models({})


def delete_from_modeldb(key):
    data = load_models()
    data.pop(key, None)
    write_models(data)
    logging.info(key + " deleted.")


def update_modeldb(key, value):
    data = load_models()
    data[key] = value
    write_models(data)
    logging.info("Data added.")