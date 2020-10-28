import pandas as pd
import csv


data = pd.read_csv("merged_scraping_results.csv", header = None)
#print(data.T.to_dict())
print(data)
