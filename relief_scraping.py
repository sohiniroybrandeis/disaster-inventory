import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time

def get_all_links(main_page_url, total_pages=20):
    links = []

    for p in range(0, total_pages):
        page_url = f"{main_page_url}&page={p}"
        print(f"Retrieving: {page_url}")
        response = requests.get(page_url)

        if response.status_code != 200:
            print("Error retrieving page")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        for header in soup.find_all("header", class_="rw-river-article__header"): # only from <header class='rw-river-article__header'>.
            a_tag = header.find("a", href=True)  # find the first <a> inside header
            if a_tag:
                full_url = urljoin(main_page_url, a_tag["href"])
                # print(full_url)
                links.append(full_url)

        time.sleep(1)  # rest time

    return links

def scrape_article_content(article_url):
    response = requests.get(article_url)
    time.sleep(1)  # rest time

    if response.status_code != 200:
        print("Error retrieving page")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.title.text.strip() if soup.title else "No Title"

    content_div = soup.find("div", {"id": "overview-content"}) # page content within <div id="overview content">
    content = content_div.get_text(separator=" ").strip() if content_div else "No Content"

    return {"title": title, "content": content, "url": article_url}

def save_to_csv(data, filename="scraped_articles.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False) #saving to csv
    print(f"Saved {len(data)} articles to {filename}")


MAIN_PAGE_URL = "https://reliefweb.int/disasters?advanced-search=%28C16.C19.C36.C41.C46.C47.C49.C54.C55.C65.C69.C75.C82.C84.C85.C223.C87.C96.C98.C102.C110.C111.C131.C138.C139.C140.C144.C146.C149.C153.C154.C163.C164.C166.C174.C175.C198.C206.C208.C210.C211.C216.C217.C8657.C220.C244.C231.C235.C240.C256.C257%29"  # Replace with your actual URL
article_links = get_all_links(MAIN_PAGE_URL, total_pages=20)

articles_data = [scrape_article_content(url) for url in article_links if scrape_article_content(url) is not None]

save_to_csv(articles_data)