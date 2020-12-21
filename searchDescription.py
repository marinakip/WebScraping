import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from langdetect import detect
# from langdetect import DetectorFactory
import spacy
from spacy_langdetect import LanguageDetector
import en_core_web_sm
from sklearn.metrics.pairwise import cosine_similarity


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
words_new = re.sub(r'\\n', ' ', data['Description'].str.e)
words = re.sub(r'\W+', ' ', words_new).lower()
tokenized_words = word_tokenize(words)

stop_words = stopwords.words('english')
filtered = [word for word in tokenized_words if word not in stop_words]
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]
print("\nLemmatized text:")
print(lemmatized)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(lemmatized)

query = "Diabetes prediction"
query_vec = vectorizer.transform([query])

results = cosine_similarity(X, query_vec)
for i in results.argsort()[-10:][::-1]:
    # print(data.iloc[i, 0], "--", data.iloc[i, 1])
    print(data.iloc[i, 0])


