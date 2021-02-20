import pandas as pd
import csv
import ast
import re
import glob
import os


def convertNumber(number):
    multiplier = number[-1].lower()
    #print("NUMBER: " + number)
    if multiplier == "k":
        number = number[:-1]
        new_number = int(float(number) * 1000)
        return new_number
    elif multiplier == "m":
        number = number[:-1]
        new_number = int(float(number) * 1000000)
        return new_number
    else:
        return number


def combine_csv():
    path = r'Scraping_Results_Yearly'
    csv_list = glob.glob(path + "/**/*.csv", recursive = True)
    csv_combined = pd.concat([pd.read_csv(f) for f in csv_list])
    path = csv_list[0]
    #print(path)
    basename = os.path.basename(os.path.dirname(path))
    #print(basename)
    name = "combined_csv_{}".format(basename)
    csv_combined.to_csv(name + '.csv', index = False)
    #print("CSV COMBINED")
    return name

#data = pd.read_csv("merged_scraping_results.csv", header=None)
#data = pd.read_csv("D:\WebScraping\Scraping_Results_Diabetes_Combined\combinedAll.csv", header=None)


filename = combine_csv()

data = pd.read_csv(filename + '.csv', header=None)
#print(data.head(5))
##data = pd.read_csv("D:\WebScraping\scraping_results_diabetes_1000_NEW.csv", header=None)
#data.drop(data.tail(1).index, inplace=True)
data.fillna('NONE', inplace=True)
#print(data)

list = []
for i in range(len(data.columns)):
    for j in range(len(data)):
        cell = data[i].iloc[j]
#        print("CELL BEFORE: " +cell)
        if 'NONE' not in cell:
 #           print("CELL:" + cell)
            cell_new = ast.literal_eval(cell)
            location = cell_new['location']
            stats = cell_new['stats_list']
            contributions = cell_new['contributions']
            contributions_new = cell_new['contributions'].split()[0]
            description_text = cell_new['description']
            words = re.sub(r'\\n', ' ', str(description_text))
            cleaned = re.sub(r'\W+', ' ', words).lower()
            description = cleaned
            url_profile = cell_new['url_profile']
            info = cell_new['info_list'][1]
            #print(cleaned)
            #print(info)
  #          print("Location: " + location)
  #          print("Stats: " + str(stats))
            try:
                followers_list = stats[0]
            except IndexError:
                followers = 0

            followers = followers_list[0]
            followers = convertNumber(followers)
 #           print("Followers: " + str(followers))
            try:
                following_list = stats[1]
            except IndexError:
                following = 0

 #           print(following_list)
            following = following_list[0]
            following = convertNumber(following)
 #           print("Following: " + str(following))
            try:
                stars_list = stats[2]
            except IndexError:
                stars = 0

            stars = stars_list[0]
            stars = convertNumber(stars)
 #           print("Stars: " + str(stars))
            contributions_new = contributions_new.replace(",", "")
            contributions = convertNumber(contributions_new)
 #           print("Contributions:" + str(contributions))
  #          print("=============================")
            line = (location, followers, following, stars, contributions, description, url_profile, info)
            list.append(line)

df = pd.DataFrame(list, columns=['Location', 'Followers', 'Following', 'Stars', 'Contributions', 'Description',
                                 'Url_profile', 'Info'])
print(df)


#df.to_csv('scraping_results_cleaned.csv', index=False, header=True)
#df.to_csv('scraping_results_cleaned_diabetes_combinedAll.csv', index=False, header=True)
#df.to_csv('scraping_results_cleaned_diabetes_1000.csv', index=False, header=True)
###df.to_csv('scraping_results_cleaned_diabetes_1000_2.csv', index=False, header=True)
df.to_csv(filename + '_cleaned.csv', index=False, header=True)
#print("CSV CREATED")


