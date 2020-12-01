import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class Scraper2:
    def __init__(self, start_url):
        self.start_url = start_url

    def scrape_results_page(self, url):
        print("START SCRAPE RESULTS PAGE")
        print("SLEEP RESULTS PAGE FOR 2secs")
        time.sleep(2)
        self.start_url = url
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        webpage_response = session.get(url, timeout=2.5)
        webpage = webpage_response.content
        soup = BeautifulSoup(webpage, "html.parser")

        description_list = []
        description = soup.find_all(attrs={"class": "mb-1"})
        for dis in description:
            description_text = dis.get_text().strip()
            description_list.append(description_text)
        print(description_list)
        results_list = []
        for link in soup.find_all("a", class_="v-align-middle"):
            new_link = "https://github.com" + str(link.get('href'))
            results_list.append(new_link)

        stars_list = []
        stars = soup.find_all(attrs={"class": "muted-link"})
        for star in stars:
            stars_text = star.get_text().split()
            stars_list.append(stars_text)

        info_list = []
        info = soup.find_all("div", class_="mr-3")
        for info in info:
            info_list.append(info.get_text().strip())
        print(info_list)

        page_links_list = []
        page_links = soup.find_all("div", attrs={"class": "d-flex d-md-inline-block pagination"})
        try:
            pages = page_links[0].find_all("a")
        except IndexError:
            print("No results")
            last_page = None
            next_page = None
            results_list = []
            page_links_list = []
            return results_list, next_page, last_page, page_links_list
        for i in pages:
            url = i.get('href')
            new_url = "https://github.com" + str(url)
            page_links_list.append(new_url)
        next_page = page_links_list[-1]  # last link is page2
        last_page = page_links_list[-2]
        return results_list, next_page, last_page, page_links_list, info_list

    def scrape_repositories(self, url):
        print("START SCRAPE RESPOSITORIES")
        print("SLEEP REPOSITORIES FOR 2secs")
        time.sleep(2)
        self.start_url = url
        repository_response = requests.get(url)
        repository = repository_response.content
        soup2 = BeautifulSoup(repository, "html.parser")

        #article = soup2.find("p", attrs={"class": "f4 mb-3"})
        article = soup2.find("article", attrs = {"class": "markdown-body entry-content container-lg"})
        article_list = []
        try:
            text = article.get_text()
        except AttributeError:
            text = "NONE"
        article_list.append(text)
        print(article_list)

        profile = soup2.find("a", attrs={"class": "url fn"})
        try:
            url = profile.get('href')
        except AttributeError:
            time.sleep(60)
            self.scrape_results_page(url)

        profile_new = "https://github.com" + str(url)

        stars = soup2.find(attrs={"class": "link-gray no-underline mr-3"})
        try:
            stars_repo = stars.get_text().strip()
        except AttributeError:
            stars_repo = "NONE"

        forks = soup2.find(attrs={"class": "link-gray no-underline"})
        try:
            forks_repo = forks.get_text().strip()
        except AttributeError:
            forks_repo = "NONE"
        return profile_new, stars_repo, forks_repo, article_list

    def scrape_profile(self, url):
        print("START SCRAPE PROFILE")
        print("SLEEP PROFILE FOR 2secs")
        time.sleep(2)
        self.start_url = url
        profile_response = requests.get(url)
        profile_page = profile_response.content
        soup2 = BeautifulSoup(profile_page, "html.parser")
        try:
            location = (soup2.find(attrs={"class": "p-label"})).text
        except AttributeError:
            location = "NONE"
        try:
            name = (soup2.find(attrs={"class": "p-name vcard-fullname d-block overflow-hidden"})).text
        except AttributeError:
            name = "NONE"
        try:
            nickname = (soup2.find(attrs={"class": "p-nickname vcard-username d-block"})).text
        except AttributeError:
            nickname = "NONE"
        try:
            bio = (soup2.find(attrs={"class": "p-note user-profile-bio mb-3 js-user-profile-bio f4"})).text
        except AttributeError:
            bio = "NONE"

        stats = soup2.find_all(attrs={"class": "link-gray no-underline no-wrap"})
        stats_list = []
        for stat in stats:
            text = stat.get_text().strip().split("\n")
            stats_list.append(text)
        try:
            contributions = (soup2.find(attrs={"class": "f4 text-normal mb-2"})).text.strip()
        except AttributeError:
            contributions = "NONE"

        profile_info = name + "\n" + nickname + "\n" + bio + "\n" + location + "\n" \
                       + str(stats_list) + "\n" + str(contributions)
        return profile_info, location, stats_list, contributions

    def scrape_page(self, start_url):
        print("START SCRAPE PAGE")
        print("SLEEP PAGE FOR 2secs")
        time.sleep(2)
        self.start_url = start_url
        results_list, next_page, last_page, pages_links, info_list = self.scrape_results_page(start_url)
        final_results_list = []
        addresses = []
        length = len(results_list)
        for i in range(length):
            repository_url = results_list[i]
            profile_url, stars_repo, forks_repo, article_list = self.scrape_repositories(repository_url)
            profile_info, location, stats_list, contributions = self.scrape_profile(profile_url)
            addresses.append(location)
            dictionary = {
                'location': location,
                'stats_list': stats_list,
                'contributions': contributions,
                'description': article_list,
                'url_profile': profile_url,
                'info_list': info_list }
            final_results_list.append(dictionary)
            print(dictionary)
        return final_results_list, next_page, last_page, addresses, pages_links



