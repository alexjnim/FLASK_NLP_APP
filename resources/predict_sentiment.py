import pandas as pd
import nltk
from afinn import Afinn
import utilities.text_normalizer as tn
from utilities.url_text_extractor import check_for_url

def predict_sentiment(request):
    """
    Predicts the sentiment of the text using Afinn library
    Input
    ----------
    request variable: Flask request variable containing the text for the article
    Returns
    ----------
    An article's sentiment
    """
    text = request.form['text']
    text = check_for_url(text)
    text = text.replace("\n", " ").replace(".  ", ". ").replace("\'", " ").strip()

    l = [text]
    # normalize the text
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')

    norm_sentences = []
    for i in range(len(l)):
        sentence = l[i]
        norm_sentence = tn.normalize_corpus(corpus=pd.Series(sentence), html_stripping=True,
                                         contraction_expansion=True, accented_char_removal=True,
                                         text_lower_case=True, text_lemmatization=True,
                                         text_stemming=False, special_char_removal=True,
                                         remove_digits=True, stopword_removal=True,
                                         stopwords=stopword_list)
        norm_sentences.append(norm_sentence[0])
    text = norm_sentences[0]
    # use Afinn to find sentiment score
    afn = Afinn()
    sentiment_score = afn.score(text)
    if sentiment_score > 0:
        sentiment = "positive"
    elif sentiment_score < 0:
        sentiment = "negative"
    elif sentiment_score == 0:
        sentiment = 'neutral'

    return sentiment
