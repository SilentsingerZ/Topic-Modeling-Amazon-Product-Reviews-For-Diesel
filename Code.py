#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:17:13 2019

@author: Freddie
"""

# =============================================================================
# Import Packages
# =============================================================================
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
#nltk.download('stopwords')
import pyLDAvis
#!pip install pyldavis
import pyLDAvis.sklearn
import numpy as np
import pandas as pd
import spacy, gensim
from collections import defaultdict

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn

# =============================================================================
# Code that looks through the Amazon product metadata and extracts ASINs
# =============================================================================

count = 0
loadedjson = open('meta_Clothing_Shoes_and_Jewelry.json', 'r')
allproducts = {}
listofcategories = {}

for aline in loadedjson:
    count += 1
    if count % 100000 == 0:
        print(count)
    aproduct = eval(aline)
    
    allproducts[aproduct['asin']] = aproduct

    for categories in aproduct['categories']:
        for acategory in categories:
            if acategory in listofcategories:
                listofcategories[acategory] += 1
            if acategory not in listofcategories:
                listofcategories[acategory] = 1

count = 0
alldieselasins = set()

for aproduct in allproducts:
    theproduct = allproducts[aproduct]
    count += 1
    if count % 100000 == 0:
        print(count/1503384)
    for categories in theproduct['categories']:
        for acategory in categories:
            if 'diesel' in acategory.lower():
                alldieselasins.add(theproduct['asin'])
                #print(theproduct['categories'])
    
f = open('asins.txt','w')
f.write(', '.join(alldieselasins))
f.close()

# =============================================================================
# Code that extracts the review for your brand
# =============================================================================

loadedjson2 = open('reviews_Clothing_Shoes_and_Jewelry.json', 'r')

count = 0
allreviews = {}
reviewers = []

for aline in loadedjson2:
    count += 1
    if count % 100000 == 0:
        print(count)
    areview = eval(aline)
    theasin = areview['asin']
    thereviewer = areview['reviewerID']
    reviewers.append(thereviewer)
    
    if theasin in alldieselasins:
        thekey = '%s.%s' % (theasin, thereviewer)
        allreviews[thekey] = areview
        
        
#output all reviews for diesel
json.dump(allreviews, open('alldieselreviews.json', 'w'))

# =============================================================================
# Code that segmenting the data
# =============================================================================
topreviewers = defaultdict(int)
for reviewer in reviewers:
    topreviewers[reviewer] += 1
topreviewers = dict(topreviewers)
# 0.95 quantile also referred to as the 95th percentile is the value such that 95% of all the reviewers fall below that value.
np.quantile(pd.DataFrame(list(topreviewers.values())), 0.95) # 5.0
topreviewerlist = []
for areviewer in topreviewers:
    reviewnumbers = topreviewers[areviewer]
    if reviewnumbers > 5.0:
        topreviewerlist.append(areviewer)

# =============================================================================
# Code that preprocesses the data (e.g., gets it ready to be topic modeled)
# =============================================================================        
df = pd.read_json('alldieselreviews.json').transpose()
df['asinandreviewText'] = df[['asin', 'reviewText']].apply(lambda x: ' '.join(x), axis=1)
df = df[df['reviewerID'].isin(topreviewerlist)]

# Convert to list
data = df.asinandreviewText.values.tolist()

# Tokenize and Clean-up using gensim’s simple_preprocess()
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
data_words = list(sent_to_words(data))

# Lemmatization
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# =============================================================================
# Code that executes and prints your topics
# =============================================================================

# Create the Document-Word matrix
stop_words = stopwords.words('english')
stop_words.append('diesel')
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words=stop_words,             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                            )
data_vectorized = vectorizer.fit_transform(data_lemmatized)

# Build LDA model with sklearn
lda_model = LatentDirichletAllocation(n_topics=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes

# GridSearch the best LDA model
# Define Search Param
search_params = {'n_components': [15], 'learning_decay': [.5, .6, .7, .8, .9, 1.0]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# How to visualize the LDA model with pyLDAvis?
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(panel, 'pyBestLDAvis.html')

# =============================================================================
# Code that executes and prints your topics
# =============================================================================
TOPICS_FILE = 'lda_topics.txt'
topics = []
def write_topics_file():
    print('Writing topics to: %s' % TOPICS_FILE)
    with open(TOPICS_FILE, 'w') as topics_file:
        lexicon = vectorizer.get_feature_names()
        n_top_words = 5
        for topic_idx, topic in enumerate(best_lda_model.components_):
            topic = topic.argsort()[:-n_top_words -1:-1]
            feature_names = []
            for i in topic:
                feature_names.append(lexicon[i])
            topic_string = " ".join(feature_names)
            topics_file.write(topic_string + '\n')
            print('Topic %d: %s' % (topic_idx, topic_string))
            topics.append(topic_string)
    print('.. done writing topics file')
write_topics_file()

# =============================================================================
# Code that classifies and separates documents by topic
# =============================================================================
CLASSIFIED_REVIEWS_FILE = 'classified_reviews.jsonl'

def write_classified_docs():
    print('Writing classified reviews to: %s' % CLASSIFIED_REVIEWS_FILE)
    with open(CLASSIFIED_REVIEWS_FILE, 'w') as outfile:
        for review in data:
            Y = vectorizer.transform([review])
            prediction = best_lda_model.transform(Y)
            predictions_list = prediction.tolist()[0]
            high_score = max(predictions_list)
            topic_id = predictions_list.index(high_score)
            outfile.write('%s\n' % json.dumps({
                'review': review,
                'topic': {
                    'id': topic_id,
                    'label': topics[topic_id]
                }
            }))
    print('.. done')
write_classified_docs()
