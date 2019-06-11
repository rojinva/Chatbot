def important_phrases(txt):
    import textrank
    return textrank.extract_key_phrases(txt)