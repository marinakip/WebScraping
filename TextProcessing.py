import pandas as pd
import nltk
data = pd.read_csv("scraping_results_cleaned_diabetes_1000.csv")
print(data.head(10))
#print(data['Description'])
print(data['Info'])
