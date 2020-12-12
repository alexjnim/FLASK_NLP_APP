import re
import numpy as np
import pandas as pd
import pickle
import nltk
import networkx
import resources.text_normalizer as tn
from sklearn.feature_extraction.text import TfidfVectorizer



def normalize_document(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def normalize_sentences(sentences):
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')

    norm_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        norm_sentence = tn.normalize_corpus(corpus=pd.Series(sentence), html_stripping=True,
                                         contraction_expansion=True, accented_char_removal=True,
                                         text_lower_case=True, text_lemmatization=True,
                                         text_stemming=False, special_char_removal=True,
                                         remove_digits=True, stopword_removal=True,
                                         stopwords=stopword_list)
        norm_sentences.append(norm_sentence[0])

    return np.array(norm_sentences)

def tfidf_matrix(norm_sentences):
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences)
    dt_matrix = dt_matrix.toarray()
    return dt_matrix

# +
def summarize_text(request):
    text = request.form['text']

#     text = text.replace("\n", ". ").replace("\n\n", " ").replace("\'", " ").strip()
    text.replace(". \n", ". ").replace("\n\n", " ").replace("\n", " a. ").replace("\'", " ").strip()

    sentences = nltk.sent_tokenize(text)

    to_remove = []
    for i in range(len(sentences)):
        if sentences[i][-2:] == 'a.':
            to_remove.append(sentences[i])
            continue

        if sentences[i][-1] == '?':
            to_remove.append(sentences[i])
            continue
    sentences = [x for x in sentences if x not in to_remove]

    if len(sentences) <= 3:
        num_sentences = 1
    elif len(sentences) > 3 & len(sentences) <= 15:
        num_sentences = 3
    elif len(sentences) > 15:
        num_sentences = np.floor(len(sentences)/5)

    normalize_corpus = np.vectorize(normalize_document)
    norm_sentences = normalize_corpus(sentences)
    #norm_sentences = normalize_sentences(sentences)
    print(norm_sentences)
    dt_matrix = tfidf_matrix(norm_sentences)

    similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()

    summary = np.array(sentences)[top_sentence_indices]

    for i in range(len(summary)):
        summary[i] = summary[i].replace(' s ', '\'s ')

    return summary
