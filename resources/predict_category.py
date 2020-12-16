import numpy as np
import pandas as pd
import pickle
import nltk
import resources.text_normalizer as tn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC

def predict_category(request):
    text = request.form['text']
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    norm_corpus = tn.normalize_corpus(corpus=pd.Series(text), html_stripping=True,
                                     contraction_expansion=True, accented_char_removal=True,
                                     text_lower_case=True, text_lemmatization=True,
                                     text_stemming=False, special_char_removal=True,
                                     remove_digits=True, stopword_removal=True,
                                     stopwords=stopword_list)

    model_path = './CLASSIFIER/model/best_linear_cvs.pkl'
    loaded_model = pickle.load(open(model_path, 'rb'))
    results = loaded_model.predict(np.array(norm_corpus))

    return results[0]
