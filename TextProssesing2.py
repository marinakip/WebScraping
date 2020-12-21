import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from geopy import Nominatim


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
english = set(nltk.corpus.words.words())



def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


def clean_text(text):
    words = re.sub(r'\\n', ' ', text)
    cleaned = re.sub(r'\W+', ' ', words).lower()
    tokenized = word_tokenize(cleaned)
    filtered = [word for word in tokenized if word not in stop_words and word in english]
    normalized = " ".join([lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in filtered])
    return normalized


data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
cleaned_text = [clean_text(text) for text in data['Description']]
#print(cleaned_text)


vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(cleaned_text)
#print(tfidf_scores)

query = "glucose meter"
query_vec = vectorizer.transform([query])

results = cosine_similarity(tfidf_scores, query_vec)

#print(len(results))
#print(len(data['Description']))


results_nonzero = np.nonzero(results)[0]

geocoding_results = []

print("Length Result Addresses: {}".format(len(results_nonzero)))

for i in results_nonzero:
    address = data.iloc[i, 0]
    locator = Nominatim(user_agent="myGeocoder")
    followers = data.iloc[i, 1]
    # print(address)
    # print(followers)
    following = data.iloc[i, 2]
    stars = data.iloc[i, 3]
    contributions = data.iloc[i, 4]
    # description = data.iloc[i, 5]
    url_profile = data.iloc[i, 6]
    # info

    try:
        location = locator.geocode(address, timeout = 30)
        latitude = location.latitude
        longitude = location.longitude
        location = locator.reverse([latitude, longitude])
        #print(location.raw)
        display_name = location.raw['display_name']
        country = location.raw['address']['country']
        country_code = location.raw['address']['country_code']
        #print(location.raw)
    except AttributeError:
        # location = "NONE"
        latitude = None
        longitude = None

    except KeyError:
        country = None
        country_code = None

    finally:
        # print("Display Name = {}".format(display_name))
        # print("Latitude = {}, Longitude = {}".format(latitude, longitude))
        # print("Country = {}, Country Code = {}".format(country, country_code))

        dictionary = {
            'Address': address,
            'Latitude': latitude,
            'Longitude': longitude,
            'Country': country,
            'Country_Code': country_code,
            'Followers'	: followers,
            'Following'	: following,
            'Stars'	: stars,
            'Contributions'	: contributions,
            #'Description': description
            'Url_profile'	: url_profile
            #Info : info

        }
        geocoding_results.append(dictionary)

df = pd.DataFrame(geocoding_results)
print("Length Dataframe: {}".format(df.size))
print(df)

df.to_csv('search_results.csv')


# print(vectorizer.vocabulary_)
# print(vectorizer.get_feature_names())
# print(tfidf_scores)

# for i in results.argsort()[-10:][::-1]:
#     print(data.iloc[i,0],"--", data.iloc[i,1])


# feature_names = vectorizer.get_feature_names()
# print(feature_names)
#
# words_index = [f"Description {i+1}" for i in range(len(data['Description']))]
#
# # create pandas DataFrame with tf-idf scores
# try:
#   df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=words_index)
#   print(df_tf_idf.head(10))
# except:
#   pass
