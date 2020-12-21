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

# search_term = input("Search: ")
# print("You searched for: " + search_term)

data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech




#description_lemmatized = []
#============================================================================
# for index, row in data.iterrows():
#     # print("ROW DESCRIPTION")
#     # print(row['Description'])
#     # print("AFTER 1")
#     words_new = re.sub(r'\\n', ' ', row['Description'])
#     # print(words_new)
#     # print("AFTER 2")
#     words = re.sub(r'\W+', ' ', words_new).lower()
#     # print(words)
#     tokenized_words = word_tokenize(words)
#
#     stop_words = stopwords.words('english')
#     filtered = [word for word in tokenized_words if word not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]
#     print("\nLemmatized text:")
#     print(lemmatized)
#     data['New Description'] = lemmatized
#     print("\nNew Description:")
#     print(data['New Description'])
#     # for word in lemmatized:
#     #     description_lemmatized.append(word)
#
# # data['New Description'] = description_lemmatized
# print("ALL")
# print(data['New Description'])
# # print(description_lemmatized)
#=============================================================================
#     stemmer = PorterStemmer()
#     stemmed = [stemmer.stem(token) for token in tokenized_words]
#     # print("Stemmed text:")
#     # print(stemmed)
#
#     lemmatizer = WordNetLemmatizer()
#     #lemmatized = [lemmatizer.lemmatize(token, pos_tag(token)) for token in tokenized_words]
#     lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]
#     #lemmatized2 = [lemmatizer.lemmatize(token) for token in filtered]
#
#     # print("\nLemmatized text:")
#     # print(lemmatized)
#
#    # description_list = [description_list.append(word) for word in lemmatized]
#
#     #
#     # nlp = en_core_web_sm.load()
#     # nlp.add_pipe(LanguageDetector(), name = 'language_detector', last = True)
#     # doc = nlp(lemmatized)
#     # print(doc._.language) # {'language': 'en', 'score': 0.9999978351575265}
#
#     #
#     #
#     # for word in lemmatized:
#     #     print(word)
#     #     if not re.match(r'\d+', word):
#     #         # print("on")
#     #         nlp = en_core_web_sm.load()
#     #         nlp.add_pipe(LanguageDetector(), name = 'language_detector', last = True)
#     #         doc = nlp(word)
#     #         print(doc._.language)  # {'language': 'en', 'score': 0.9999978351575265}
#
#     #print(lemmatized2)
#
#
#
#     # print("\nBag of Words:")
#     # bag_of_words = Counter(lemmatized)
#     # print(bag_of_words)
#     #
#     # bag_of_words_creator = CountVectorizer()
#     # bag_of_words = bag_of_words_creator.fit_transform(lemmatized)
#
#     # print("===========================================")
#
#     # ngrams = ngrams(lemmatized, 2)
#     # ngrams_frequency = Counter(ngrams)
#     # print("\n Most Common:")
#     # print(ngrams_frequency.most_common(5))
#     # print("\n\n")
#
#     # # creating the bag of words LDA model
#     # lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
#     # lda_bag_of_words = lda_bag_of_words_creator.fit_transform(bag_of_words)
#
#
#     # print("~~~ Topics found by bag of words LDA ~~~")
#     # for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
#     #   message = "Topic #{}: ".format(topic_id + 1)
#     #   message += " ".join([bag_of_words_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
#     #   print(message)
#
#     # creating the tf-idf model
#     #tfidf_creator = TfidfVectorizer(min_df = 0.2)
#
#     #
#     # tfidf_creator = TfidfVectorizer()
#     # tfidf = tfidf_creator.fit_transform(filtered)
#     #
#     # # creating the tf-idf LDA model
#     # lda_tfidf_creator = LatentDirichletAllocation(learning_method = 'online', n_components = 10)
#     # lda_tfidf = lda_tfidf_creator.fit_transform(tfidf)
#     #
#     # print("\n\n~~~ Topics found by tf-idf LDA ~~~")
#     # for topic_id, topic in enumerate(lda_tfidf_creator.components_):
#     #     message = "Topic #{}: ".format(topic_id + 1)
#     #     message += " ".join([tfidf_creator.get_feature_names()[i] for i in topic.argsort()[:-5:-1]])
#     #     print(message)
#
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data['Description_New'])
# query = "diabetes prediction"
# query_vec = vectorizer.transform([query])
# results = cosine_similarity(X, query_vec)
# for i in results.argsort()[-10:][::-1]:
#     print(data.iloc[i, 0], "--", data.iloc[i, 5], "--", data.iloc[i, 6])
#
#
#
