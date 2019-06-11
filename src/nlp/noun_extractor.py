
def noun_extractor(text, corenlp):
    custom_properties = {
        'annotators': 'ner',
        'outputFormat': 'json',
        'timeout': 1000
    }
    res = corenlp.annotate(text, properties=custom_properties)

    nouns = []
    import re
    for sentence in res["sentences"]:
        for token in sentence['tokens']:
            if re.match("NN*", token['pos']):
                nouns.append(token['word'])

    return nouns