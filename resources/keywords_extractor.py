import utilities.text_normalizer as tn
import nltk
from operator import itemgetter
from gensim.summarization import keywords
from utilities.url_text_extractor import check_for_url


def get_key_words(request):

    text = request.form["text"]
    text = check_for_url(text)
    text = (
        text.replace(". \n", ". ")
        .replace("\n\n", " ")
        .replace("\n", "a. ")
        .replace("'", "'")
        .strip()
    )
    key_words = keywords(text, ratio=1.0, scores=True, lemmatize=True)
    top_key_words = [item for item, score in key_words][:5]

    top_key_words = ", ".join(top_key_words)

    return top_key_words