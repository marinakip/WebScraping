import pandas as pd
import csv
import numpy as np

data = pd.read_csv("addresses_all.csv", header = None)
#print(data)

df = data.drop(data.columns[-1], axis=1)
#print(df)

# size = data.size
# print(size)
# size2 = df.size
# print(size2)

addresses = []
counter = 0
#print(df.values)
for i in range(len(df.columns)):
    for j in range(len(df)):
        if (df[i].iloc[j] != 'NONE'):
            addresses.append(df[i].iloc[j])
            counter += 1

#print("count None: ", str(counter))
#print(addresses)
#print(df.values.size)

with open("addresses_cleaned.csv", "w", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\n")
    w.writerow(addresses)
