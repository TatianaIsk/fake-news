from array import array
from typing import Any

import numpy as np
import pandas as pd
import sklearn.metrics._plot.precision_recall_curve
from pandas import DataFrame
from pandas._testing import at
from pandas.compat.numpy import function
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
import numpy

df = pd.read_table('predictions.tsv')

df.head(5)

df['is_fake'] = df['is_fake'].apply(int)

df.head(5)

df['is_fake'].value_counts()

for c in df[df['is_fake'] == 1]['title'].head(5):
    print(c)

for c in df[df['is_fake'] == 0]['title'].head(5):
    print(c)

predictions_df: object
(predictions_df,) = train_test_split(df, test_size=500)
var = predictions_df.shape
predictions_df['is_fake'].value.couts()
predictions_df['is_fake'].value.couts()

sentence_example: object = df.iloc[1]['title']
tokens = word_tokenize(sentence_example, language='russian')
tokens_without_punctuation: list[str] = [i for i in tokens if i not in string.punctuation]
russian_stop_words = stopwords.words('russian')
tokens_without_stop_words_and_punctuation = [i for i in tokens_without_punctuation if i not in russian_stop_words]
snowball = SnowballStemmer(language='russian')
stemmed_tokens = [snowball.stem(i) for i in tokens_without_stop_words_and_punctuation]

print(f"Text: {sentence_example}")
print(f"--------")
print(f"Tokens: {tokens}")
print(f"--------")
print(f"Tokens without punctuation: {tokens_without_punctuation}")
print(f"--------")
print(f"Tokens without punctuation and stop words: {tokens_without_stop_words_and_punctuation}")
print(f"--------")
print(f"After stemmed: {stemmed_tokens}")

snowball = SnowballStemmer(language='russian')
russian_stop_words = stopwords.words('russian')


def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens: list[str] = word_tokenize(sentence, language='russian')
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

tokenize_sentence(sentence_example)

vectorizer = TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))
features = vectorizer.fit_transform(['title'])

model = LogisticRegression(random_state=0)
model.fit(features, ['is_fake'])

LogisticRegression(random_state=0)

model.predict(features[0])
array([0])

var = ['title'].iloc[0]

model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ('model', LogisticRegression(random_state=0))
]
)

model_pipeline.fit(['title'], ['is_fake'])

precision_score(y_true=predictions_df['is_fake'], y_pred=model_pipeline.predict(predictions_df['title']))
recall_score(y_true=predictions_df['is_fake'], y_pred=model_pipeline.predict(predictions_df['title']))

prec, rec, thresholds = precision_recall_curve(y_true=predictions_df['is_fake'], probas_pred=model_pipeline.predict_proba(predictions_df['title']))
plot_precision_recall_curve(estimator=model_pipeline, X=predictions_df['title'], y=predictions_df['is_fake'])

np.where(prec > 0.95)
var = thresholds[374]
precision_score(y_true=predictions_df['is_fake'], y_pred=model_pipeline.predict_proba(predictions_df['title'])[:, 1] > thresholds[374])
recall_score(y_true=predictions_df['is_fake'], y_pred=model_pipeline.predict_proba(predictions_df['title'])[:, 1] > thresholds[374])

grid_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ('model',
     GridSearchCV(
         LogisticRegression(random_state=0),
         param_grid={'C': [0.1, 1, 10.]},
         cv=3,
         verbose=4
     )
     )
])

grid_pipeline.fit(predictions_df['title'], predictions_df['is_fake'])