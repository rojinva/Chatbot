#!/usr/bin/env bash

# Create a new virtual environment
conda create -n rasa numpy scipy scikit-learn matplotlib jupyter python=2.7
source activate rasa

# Install Rasa dependencies
pip install -r rasa_nlu/dev-requirements.txt

# Install necessary python package
pip install pycorenlp nltk textblob rasa_nlu==0.9.0
conda install spacy
python -m easy_install lib/DS_Utilities-0.1-py2.7.egg
pip install git+http://github.com/davidadamojr/TextRank.git

# Corpus data for NLTK, Spacy and Textblob
python -m textblob.download_corpora
python -m nltk.downloader all

python -m spacy download en

#python -m spacy download en_core_web_md
#sputnik --name spacy --repository-url http://index.spacy.io install en==1.1.0

# More conda dependencies
while read requirement; do conda install --yes $requirement; done < package-requirement.txt
