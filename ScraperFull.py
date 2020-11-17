import time
import pendulum
import pandas as pd
from Scraper import Scraper

def create_start_url(year1, month1, day1, year2, month2, day2):
    url_list = []
    #query = 'diabetes'
    query = 'machine+learning'
    start = pendulum.datetime(year1, month1, day1)
    end = pendulum.datetime(year2, month2, day2)
    period = pendulum.period(start, end)

    weeks_list = []
    for dt in period.range('weeks'):
        start_week = dt.start_of('week').to_date_string()
        end_week = dt.end_of('week').to_date_string()
       #print("START WEEK: " + start_week)
       #print("END WEEK: " + end_week)
        week_tuple = (start_week, end_week)
        weeks_list.append(week_tuple)
        # print(date)

    print(weeks_list)

    for date in range(len(weeks_list)):
        start_date = weeks_list[date][0]
        end_date = weeks_list[date][1]

        #for i in range(len(start_date)):
        start_url = "https://github.com/search?utf8=%E2%9C%93&q={}+created%3A{}..{}&type=Repositories" \
                .format(query, start_date, end_date)
        #print(start_url)
        url_list.append(start_url)

    return url_list

#link_list = create_start_url(2015, 1, 1, 2016, 1, 1) #TODO FIX FOR ONE PAGE RESULTS
#link_list = create_start_url(2016, 1, 1, 2017, 1, 1)
#link_list = create_start_url(2017, 1, 1, 2018, 1, 1)
#link_list = create_start_url(2018, 1, 1, 2019, 1, 1)
#link_list = create_start_url(2019, 1, 1, 2020, 1, 1)
#link_list = create_start_url(2020, 1, 1, 2020, 11, 1)
#link_list = create_start_url(2020, 5, 25, 2020, 11, 1)
link_list = create_start_url(2016, 7, 11, 2017, 1, 1)

#print(link_list)

#count = 1
count = 28
for item in range(len(link_list)):
    start_url = link_list[item]
    print(start_url)
    scraper = Scraper(start_url)
    #last_page = scraper.scrape_page(start_url)[2]
    #print("last page: "+str(last_page))

    counter = 1

    print("SCRAPING BATCH NUMBER " + str(count))
    print("SLEEP SCRAPING BATCH FOR 3secs")
    time.sleep(3)
    try:
        final_results_list, next_page, last_page, addresses, pages_links = scraper.scrape_page(start_url)
    except ConnectionError:
        print("Connection Error")
        time.sleep(30)
        final_results_list, next_page, last_page, addresses, pages_links = scraper.scrape_page(start_url)
        continue
    print(pages_links)
    if next_page is not None:
        all_final_results_list = [final_results_list]  # results of one page
        final_addresses = [addresses]
        print("OK")
        while next_page != last_page:
            counter += 1
            print("SCRAPING PAGE " + str(count))
            new_final_results_list, new_next_page, last_page, addresses, pages_links = scraper.scrape_page(next_page)
            final_addresses.append(addresses)
            all_final_results_list.append(new_final_results_list)
            next_page = new_next_page
            print("OK")
        print("SCRAPING LAST PAGE")
        last_results_list, next_page, last_page, addresses, pages_links = scraper.scrape_page(last_page)
        final_addresses.append(addresses)
        all_final_results_list.append(last_results_list)

        print("all results length: " + str(len(all_final_results_list)))

        dataframe = pd.DataFrame(all_final_results_list)
        dataframe.to_csv('scraping_results_new{}.csv'.format(count), index=False, header=False)
        print("CSV FINAL CREATED")
    count += 1
