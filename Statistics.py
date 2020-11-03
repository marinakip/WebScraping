import pandas as pd
import csv
import ast


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


data = pd.read_csv("merged_scraping_results.csv", header=None)
data.drop(data.tail(1).index, inplace=True)
#print(data)

list = []
for i in range(len(data.columns)):
    for j in range(len(data)):
        cell = data[i].iloc[j]
        print("CELL BEFORE: " +cell)
        if 'NONE' not in cell:
            print("CELL:" + cell)
            cell_new = ast.literal_eval(cell)
            location = cell_new['location']
            stats = cell_new['stats_list']
            contributions = cell_new['contributions']
            contributions_new = cell_new['contributions'].split()[0]
            print("Location: " + location)
            print("Stats: " + str(stats))
            try:
                followers_list = stats[0]
            except IndexError:
                followers = 0

            followers = followers_list[0]
            followers = convertNumber(followers)
            print("Followers: " + str(followers))
            try:
                following_list = stats[1]
            except IndexError:
                following = 0

            print(following_list)
            following = following_list[0]
            following = convertNumber(following)
            print("Following: " + str(following))
            try:
                stars_list = stats[2]
            except IndexError:
                stars = 0

            stars = stars_list[0]
            stars = convertNumber(stars)
            print("Stars: " + str(stars))
            contributions_new = contributions_new.replace(",", "")
            contributions = convertNumber(contributions_new)
            print("Contributions:" + str(contributions))
            print("=============================")
            line = (location, followers, following, stars, contributions)
            list.append(line)

df = pd.DataFrame(list, columns=['Location', 'Followers', 'Following', 'Stars', 'Contributions'])
print(df)

df.to_csv('scraping_results_cleaned.csv', index=False, header=True)
print("CSV CREATED")

