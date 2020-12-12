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


data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")

for index, row in data.iterrows():
    print("ROW DESCRIPTION")
    print(row['Description'])
    print("AFTER 1")
    words_new = re.sub(r'\\n', ' ', row['Description'])
    print(words_new)
    print("AFTER 2")
    words = re.sub(r'\W+', ' ', words_new).lower()
    print(words)



    tokenized_words = word_tokenize(words)

    stop_words = stopwords.words('english')
    filtered = [word for word in tokenized_words if word not in stop_words]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokenized_words]

    lemmatizer = WordNetLemmatizer()
    #lemmatized = [lemmatizer.lemmatize(token, pos_tag(token)) for token in tokenized_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in filtered]
    print("Stemmed text:")
    print(stemmed)
    print("\nLemmatized text:")
    print(lemmatized)

    print("\nBag of Words:")
    bag_of_words = Counter(lemmatized)
    print(bag_of_words)

    bag_of_words_creator = CountVectorizer()
    bag_of_words = bag_of_words_creator.fit_transform(lemmatized)

    print("===========================================")

    # ngrams = ngrams(lemmatized, 2)
    # ngrams_frequency = Counter(ngrams)
    # print("\n Most Common:")
    # print(ngrams_frequency.most_common(5))
    # print("\n\n")

    # creating the bag of words LDA model
    lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
    lda_bag_of_words = lda_bag_of_words_creator.fit_transform(bag_of_words)


    print("~~~ Topics found by bag of words LDA ~~~")
    for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
      message = "Topic #{}: ".format(topic_id + 1)
      message += " ".join([bag_of_words_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
      print(message)



