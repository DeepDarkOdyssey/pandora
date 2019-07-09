from string import punctuation
import spacy

nlp = spacy.load('en_core_web_sm')


def get_hash(salt: str = ''):
    return lambda x: hash(x + salt)


def preprocess(text: str):
    doc = nlp(text)
    new_text = ''
    prev_end_char = 0
    if len(doc.ents) > 0:
        for ent in doc.ents:
            new_text += text[prev_end_char:ent.start_char] + ent.label_
            prev_end_char = ent.end_char
        new_text += text[prev_end_char:]
    else:
        new_text = text
    doc = nlp(new_text)
    result = []
    for token in doc:
        if not token.text.strip():
            continue
        if token.text in punctuation:
            continue
        if token.text.isupper():
            result.append(token.text)
        else:
            result.append(token.lemma_)
        
    return result
