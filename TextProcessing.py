# import pandas as pd
# import nltk
# from nltk import sent_tokenize
# from nltk import word_tokenize
# #nltk.download()
#
#
# data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
# # print(data.head(10))
# # print(data['Description'])
# #sentences = sent_tokenize(data['Description'].str)
# #print(sentences[0])
# #data['Description'] = data['Description'].str.replace(r'\W', ' ')
# data['Words'] = data['Description'].apply(word_tokenize)
# #print(data['Description'])
# print(data['Words'])
# data['Sentences'] = data['Description'].apply(sent_tokenize)
# print(data['Sentences'])
# # print(data['Info'])



import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from nltk.util import ngrams

#from part_of_speech import get_part_of_speech
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
    bag_of_looking_glass_words = Counter(lemmatized)
    print(bag_of_looking_glass_words)

    print("===========================================")

    ngrams = ngrams(lemmatized, 5)
    ngrams_frequency = Counter(ngrams)
    print("\n Most Common:")
    print(ngrams_frequency.most_common(10))
    print("\n\n")


