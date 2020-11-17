import pandas as pd


#data = pd.read_csv("scraping_results_cleaned.csv", header=0)
data = pd.read_csv("scraping_results_cleaned_diabetes_combinedAll.csv", header=0)
df = data.loc[:, 'Followers':'Contributions']
print(df.head(10))
data['Weight'] = (df['Followers'] * 0.2) + (df['Following'] * 0.1) + (df['Stars'] * 0.3) + (df['Contributions'] * 0.4)
#print(df['Weight'])
#data.to_csv('scraping_results_weighted.csv', index=False, header=True)
data.to_csv('scraping_results_cleaned_diabetes_combinedAll_weighted.csv', index=False, header=True)
print("CSV CREATED")


