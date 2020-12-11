# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# according to the results from "02_model_training", Linear SVC seemed to perform the best

# +
import numpy as np
import resources.text_normalizer as tn
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

data_df = pd.read_csv('CLASSIFIER/data/cleaned_all_data.csv')

# +
from sklearn.model_selection import train_test_split

train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(
    np.array(data_df['clean text']),
                                         np.array(data_df['category label']),
                                         np.array(data_df['category']),
                                         test_size=0.30, random_state=42)
train_corpus.shape, test_corpus.shape
# -

from collections import Counter
trd = dict(Counter(train_label_names))
tsd = dict(Counter(test_label_names))
(pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
             columns=['category', 'Train Count', 'Test Count'])
.sort_values(by=['Train Count', 'Test Count'],
             ascending=False))

# ### gridsearch pipeline with linear svc and tf-idf

# +
# Tuning our Multinomial Naïve Bayes model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                        ('svm', LinearSVC())
                       ])

### here we evaluate this on bigrams and unigrams tf-idf and change the alpha value of MNB

param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'svm__C': [0.5, 1.0, 1.5],
              'svm__penalty': ['l1', 'l2']
                }

gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, verbose=2)
gs_svm = gs_svm.fit(train_corpus, train_label_names)
# -

gs_svm.best_estimator_.get_params()

cv_results = gs_svm.cv_results_
results_df = pd.DataFrame({'rank': cv_results['rank_test_score'],
                           'params': cv_results['params'],
                           'cv score (mean)': cv_results['mean_test_score'],
                           'cv score (std)': cv_results['std_test_score']}
              )
results_df = results_df.sort_values(by=['rank'], ascending=True)
pd.set_option('display.max_colwidth', 100)
results_df

best_svm_test_score = gs_svm.score(test_corpus, test_label_names)
print('Test Accuracy :', best_svm_test_score)

# ### model performance evaluation with Linear SVC

# +
import resources.model_evaluation_utils as meu

svm_predictions = gs_svm.predict(test_corpus)
unique_classes = list(set(test_label_names))
meu.get_metrics(true_labels=test_label_names, predicted_labels=svm_predictions)
# -

meu.display_classification_report(true_labels=test_label_names,
                                  predicted_labels=svm_predictions,
                                  classes=unique_classes)

# #### testing with new data

# +
test_text = 'Food and drink supplies in the UK face more disruption after the end of the Brexit transition period than they did from Covid, the industry has said. There are 14 [working] days to go, the Food and Drink Federations (FDF) chief executive, Ian Wright, told MPs. How on earth can traders prepare in this environment he added. Noting that rules for sending goods from Welsh ports to Northern Ireland had only just been published, he said: Its too late, baby. Uncertainty over a deal and new border checks would make it difficult to guarantee the movement of food through ports without delays, he said. Mr Wright was giving evidence to the Commons business committee on Brexit preparedness. He said there was a big concern that the problems would erode the confidence of shoppers in the supply chain, adding: It has done very well over Covid and shoppers will expect the same thing over Brexit, and they may not see it. PM: Sweet reason can get us to post-Brexit deal NI food supply warnings taken very seriously No clue We cant be absolutely certain about the movement of food from the EU to the UK from 1 January for two reasons, Mr Wright said. One is checks at the border. The other is tariffs, and the problem with tariffs is, we dont know what they will be. Mr Wright added: With just 14 working days to go, we have no clue whats going to happen in terms of whether we do or dont face tariffs. And that isnt just a big imposition. Its a binary choice as to whether you do business in most cases. My members will not know whether theyre exporting their products after 1 January, or whether theyll be able to afford to import them and charge the price that the tariff will dictate. Mr Wright warned that while he expected Kent and Operation Brock to work "reasonably well", he was less confident about ports such as Holyhead, with goods heading to Northern Ireland.'

test_text = 'The guitarist and producer said record labels retain up to 82 of the royalties generated from music played on services like Spotify Apple Music and Amazon Music calling the system just ridiculous And he accused the major labels of deliberately withholding money from artists I look at the record labels as my partners And the interesting thing is that every single time Ive audited my partners I find money Every single time And sometimes its staggering the amount of money Rodgers whose credits include Chics Le Freak Madonnas Like A Virgin and David Bowies Lets Dance said the industry needed to change the way streaming payments are calculated Currently each play of a song is counted as a sale which gives labels the lions share of the income Rodgers argued that a stream was more like a radio broadcast or a licence of the recording which would give artists 50 of the royalties Labels have unilaterally decided that a stream is considered a sale because it maximises their profits he said Artists and songwriters need to update clauses in their contracts to reflect the true nature of how their songs are being consumed  which is via a licence It is something that people are borrowing from the streaming services Live music eviscerated However Rodgers was optimistic that labels streaming services writers and musicians could negotiate a fairer deal and asked MPs to make the UK a leader in regulating the streaming market This can now be a great paradigm shift for songwriters and artists all over the world he said Mercurynominated jazz musician Soweto Kinch said the timing of the inquiry was particularly important after Covid19 had eviscerated the live music scene He said streaming had placed a particular strain on niche musical genres and experimental musicians because of an overriding focus on chart music Wed never have a Kate Bush or a David Bowie in todays music ecology because its very risk averse he said You are making songs for playlists you are not taking the incredible musical risks that Bowie might have taken years ago Nadine Shah IMAGE COPYRIGHTGETTY IMAGES image caption Nadine Shah said she had considered giving up music because earning a living was so difficult The inquiry continues into 2021 and will hear the perspectives of industry experts artists and record labels as well as streaming platforms themselves At the first session last month musician Nadine Shah told MPs many fellow musicians were afraid to give evidence because we do not want to lose favour with the streaming platforms and the major labels In response committee chair Julian Knight MP later warned companies against interfering with the inquiry We have been told by many different sources that some of the people interested in speaking to us have become reluctant to do so because they fear action may be taken against them if they speak in public Knight said I would like to say that we would take a very dim view if we had any evidence of anyone interfering with witnesses to one of our inquiries No one should suffer any detriment for speaking to a parliamentary committee and anyone deliberately causing harm to one of our witnesses would be in danger of being in contempt of this House This committee will brook no such interference and will not hesitate to name and shame anyone proven to be involved in such activity'


# -

pd.Series(test_text)

# +
import nltk
stopword_list = nltk.corpus.stopwords.words('english')
# just to keep negation if any in bi-grams
stopword_list.remove('no')
stopword_list.remove('not')

norm_corpus = tn.normalize_corpus(corpus=pd.Series(test_text), html_stripping=True,
                                 contraction_expansion=True, accented_char_removal=True,
                                 text_lower_case=True, text_lemmatization=True,
                                 text_stemming=False, special_char_removal=True,
                                 remove_digits=True, stopword_removal=True,
                                 stopwords=stopword_list)
# -

gs_svm.best_estimator_.predict(np.array(norm_corpus))

# Extract test document row numbers
train_idx, test_idx = train_test_split(np.array(range(len(data_df['text']))), test_size=0.30, random_state=42)

svm_predictions = gs_svm.predict(test_corpus)
test_df = data_df.iloc[test_idx]
test_df['Predicted Name'] = svm_predictions
test_df.head()

pd.set_option('display.max_colwidth', 200)
res_df = (test_df[(test_df['category'] == 'business')
                  & (test_df['Predicted Name'] == 'politics')])
res_df



# # save model

import pickle
filename = 'CLASSIFIER/model/best_linear_cvs.pkl'
pickle.dump(gs_svm.best_estimator_, open(filename, 'wb'))

# # load model

import pickle
filename = 'CLASSIFIER/model/best_linear_cvs.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

loaded_model

# +
text = 'Boris Johnson will fly to Brussels later for talks on a post-Brexit deal with the European Commission President Ursula von der Leyen. Time is running out to reach a deal before 31 December, when the UK stops following EU trading rules. The pair will hold talks over dinner, after negotiations between officials ended in deadlock. Major disagreements remain on fishing rights, business competition rules and how a deal will be policed. At the dinner, expected to begin at 19:00 GMT, Prime Minister Johnson will work through a list of the major sticking points with Mrs von der Leyen, who is representing the leaders of the 27 EU nations. A UK government source said progress at a political level may allow the negotiations - between the UK\'s Lord Frost and EU\'s Michel Barnier - to resume over the coming days.'

import nltk
stopword_list = nltk.corpus.stopwords.words('english')
# just to keep negation if any in bi-grams
stopword_list.remove('no')
stopword_list.remove('not')
norm_corpus = tn.normalize_corpus(corpus=pd.Series(text), html_stripping=True,
                                 contraction_expansion=True, accented_char_removal=True,
                                 text_lower_case=True, text_lemmatization=True,
                                 text_stemming=False, special_char_removal=True,
                                 remove_digits=True, stopword_removal=True,
                                 stopwords=stopword_list)
# -

results = loaded_model.predict(np.array(norm_corpus))

results[0]


