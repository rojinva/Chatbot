
def sentiment_extractor(text, corenlp):
    custom_properties = {
        'annotators': 'sentiment',
        'outputFormat': 'json',
        'timeout': 1000
    }

    res = corenlp.annotate(text, properties=custom_properties)
    avg_sentiment = 0
    for s in res["sentences"]:
        avg_sentiment += (int(s['sentimentValue']) - 2)

    if avg_sentiment < 0:
        avg_sentiment = "negative"
    elif avg_sentiment > 0:
        avg_sentiment = "positive"
    else:
        avg_sentiment = "neutral"

    return avg_sentiment