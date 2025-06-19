import asyncio
import polars as pl
from datetime import datetime
from playwright.async_api import async_playwright
import re
import polars as pl
import json
from bs4 import BeautifulSoup
import aiohttp
import os


schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8, "URL": pl.Utf8}

article_count = len(os.listdir("./data/articles/")) 
# general helpers
def write_article(text, person, publisher):
    global article_count
    article_count +=  1
    filename = f"{article_count}_{person}_{publisher}.json"
    with open("./data/articles/" + filename, 'w') as f: ## !!!
        json.dump(text, f, indent=2)


def get_publication_date(metadata):
    if not metadata:
        return None     
    pub_date = metadata.get("datePublished")
    if not pub_date:
        pub_date = metadata.get("dateCreated")
    if not pub_date:    
        pub_date = metadata.get("dateModified")
    if not pub_date:
        pub_date = metadata.get("dateUpdated")
    if pub_date:
        try:
            date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            return date.date()
        except ValueError:
            return None
    return None

def clean_paragraph(text):
    stop_phrases = [ "related articles", "related stories", "related news", "recommended videos", "trending now", "you may also like", "recommended for you",
                    "more stories","read next","other news","popular articles","top stories","top news","top articles","most popular articles", "most popular stories", "trending videos",
                    "trending articles","trending news","trending stories","trending now","with files from", "thanks for reading this article", "thank you for reading this article", "most commented articles"]
    for phrase in stop_phrases:
        if phrase in text.lower():
            return "stop phrase"
    
    if len(text.split()) <= 2:
        return None
    text = ' '.join(text.split())
    if "href" in text:
        text = text.split("<a href", 1)[0]
    if "©" in text:
        return None
    
    account_words = ["subscribe now","create an account","sign up for an account","sign in if you have an account","sign-in if you have an account", "subscribe to get the latest",
                     "log in if you have an account","sign in to your account","sign-in to your account","log in to your account", "type your email", "enter your email",
                     "recaptcha", "captcha", "enter your email address"]
    for word in account_words:
        if word in text.lower():
            return None

    return text


# static helpers
static_timeout = 10

async def getHTML(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=static_timeout) as response:
            return await response.text()
        
def get_static_metadata(soup):
    ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})   
    for script in ld_scripts:
        if not script.string:
            continue  
        # temp store this script 
        article_data = {}
        temp_data = json.loads(script.string)
        if isinstance(temp_data, dict):
            if temp_data.get('@type') == 'NewsArticle' or temp_data.get('@type') == 'ReportageNewsArticle':
                article_data = temp_data
        # where data saved as objs, check that we actually have NewsArticle obj
        if isinstance(temp_data, dict) and not article_data:
            continue  
        ld_data = json.loads(script.string)
        # arr
        if isinstance(ld_data, list):
            news_article = next((item for item in ld_data if item.get('@type') == 'NewsArticle'), None)
            if news_article:
                return news_article
        # obj
        else:
            return ld_data             
        return None
    
def get_static_publisher(news_article):
    pub = news_article.get("publisher", {})
    if not pub:
        pub = news_article.get("provider", {})
    if not pub:
        pub = news_article.get("source", {})
    if not pub:
        pub = news_article.get("creator", {})
    if not pub:
        pub = news_article.get("NewsMediaOrganization", {})
    if not pub:
        pub = news_article.get("author", {})
    return pub.get("name")

def extract_all_text(soup):
    paragraphs = soup.article.find_all('p')
    text = soup.title.get_text()
    for p in paragraphs:
        text += "\n"
        text += p.get_text() 
    return text.split("\n")

def add_data(publisher, url, publication_date, paras, person, event_date):
    # lines = text.split("\n")
    df = pl.DataFrame(schema=schema)
    i = 0
    for p in paras:
        cleaned_p = clean_paragraph(p)
        if cleaned_p is None:
            continue
        if cleaned_p == "stop phrase":
            df.rechunk()
            return df
        temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": publication_date, "Publisher": publisher, "URL":url, "Paragraph Index": [i], "Paragraph Text": cleaned_p})
        df = df.vstack(temp)
        i += 1
    df.rechunk()
    return df

# static extractor for articles that don't need dynamic rendering
async def static_extractor(url, person, event_date):
    try:
        html_content = await getHTML(url)
        soup = BeautifulSoup(html_content, features="lxml")
        metadata = get_static_metadata(soup)
        publisher = get_static_publisher(metadata)
        publication_date = get_publication_date(metadata)
        text = extract_all_text(soup)
        if (len(text) >= 5):
            write_article(text, person, publisher)
            return add_data(publisher, url, publication_date, text, person, event_date)
        else:
            print("Not enough text in ", url, " trying Playwright") # try playwright
    except Exception as e:
        print("Static extraction failed for ", url, " trying Playwright...")
    return await dynamic_extractor(url, person, event_date)



# dynamic helpers
async def get_dynamic_metadata(page):
    ld_scripts = await page.query_selector_all('script[type="application/ld+json"]')
    
    for script in ld_scripts:
        script_content = await script.text_content()
        if not script_content:
            continue
            
        try:
            ld_data = json.loads(script_content)
            if isinstance(ld_data, list):
                news_article = next((item for item in ld_data if item.get('@type') == 'NewsArticle' or 
                                     item.get('@type') == 'ReportageNewsArticle'), None)
                if news_article:
                    return news_article
            elif isinstance(ld_data, dict):
                if ld_data.get('@type') == 'NewsArticle' or ld_data.get('@type') == 'ReportageNewsArticle':
                    return ld_data
        except json.JSONDecodeError:
            continue
    
    return None

async def get_dynamic_publisher(page, metadata, url):
    # publisher/org/author in JSON-LD 
    if metadata and isinstance(metadata, dict):
        if isinstance(metadata.get("publisher"), dict):
            return metadata["publisher"].get("name", "Unknown")
        if metadata.get("@type") == "NewsMediaOrganization":
            return metadata.get("name", "Unknown")
        if isinstance(metadata.get("author"), list):
            for author in metadata["author"]:
                if isinstance(author, dict):
                    return author.get("name", "Unknown")
    # og:site-name meta tag
    try:
        publisher_meta = await page.query_selector('meta[property="og:site_name"]')
        if publisher_meta:
            publisher = await publisher_meta.get_attribute('content')
            if "." in publisher:
                publisher = re.split(r'(?=[A-Z])', publisher.split(".", 1)[0])
                publisher = ' '.join(filter(None, publisher))
            return publisher
    except:
        pass
                
    # article:publisher meta tag
    try:
        publisher_meta = await page.query_selector('meta[property="article:publisher"]')
        if publisher_meta:
            publisher = await publisher_meta.get_attribute('content')
            if "." in publisher:
                publisher = re.split(r'(?=[A-Z])', publisher.split(".", 1)[0])
                publisher = ' '.join(filter(None, publisher))
            return publisher
    except:
        pass

    # article:author meta tag
    try:
        publisher_meta = await page.query_selector('meta[property="article:author"]')
        if publisher_meta:
            publisher = await publisher_meta.get_attribute('content')
            if "." in publisher:
                publisher = re.split(r'(?=[A-Z])', publisher.split(".", 1)[0])
                publisher = ' '.join(filter(None, publisher))
            return publisher
    except:
        pass
    
    return "Unknown"

# general dynamic extractor to pull content from js-rendered sites
async def dynamic_extractor(url, person, event_date):
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        # browser = await p.webkit.launch(headless=False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()
        # print(f"[DEBUG] Navigating to {url} from GENERAL")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded") 

        metadata = await get_dynamic_metadata(page)
        publisher = await get_dynamic_publisher(page, metadata, url)
        published_on = get_publication_date(metadata)
        # stopped grabbing figcaptions
        # replaced if/else with try/except for safer handling when no article tag
        try:
            await page.wait_for_selector("article :is(p, h1, h2)", timeout=5000)
            paras = await page.locator('article :is(p, h1, h2)').all_text_contents()
        except:
            # print(f"[DEBUG] Fallback to generic selector for {url}")
            await page.wait_for_selector(":is(p, h1, h2)", timeout=5000)
            paras = await page.locator(':is(p, h1, h2)').all_text_contents()
        index = 0
        cleaned_paras = []
        for p in paras:
            p = clean_paragraph(p)
            if p is None:
                continue
            if p == "stop phrase":
                df.rechunk()
                await browser.close()
                write_article(cleaned_paras, person, publisher)
                return df
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
            cleaned_paras.append(p)
        df.rechunk()

        await browser.close()
        write_article(cleaned_paras, person, publisher)

    return df

# extracts content from cbc articles
async def cbc_extractor(url, person, event_date):
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        print(f"[DEBUG] Navigating to {url} from CBC")
        await page.goto(url)


        # get/set data that will be re-used first
        metadata = await get_dynamic_metadata(page)
        publisher = "CBC"
        published_on = get_publication_date(metadata)

        headline = await page.locator("h1.detailHeadline").all_text_contents() or []
        byline = await page.locator("h2.deck").all_text_contents() or []
        # !!!
        # dont get img content
        # also removed figcaption
        # img = []
        # try:
        #     img_content = await page.locator("figcaption").first.text_content()
        #     if img_content:
        #         img = [img_content]
        # except:
        #     pass        
        body = await page.locator(".story p:not(article p), .story h2:not(article h2)").all_text_contents() or []
        paras = headline + byline + body

        index = 0
        cleaned_paras = []
        for p in paras:
            p = clean_paragraph(p)
            if p is None:
                continue
            if p == "stop phrase":
                df.rechunk()
                await browser.close()
                write_article(cleaned_paras, person, publisher)
                return df
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
            cleaned_paras.append(p)

        df.rechunk()
        await browser.close()
        write_article(cleaned_paras, person, publisher)
    return df

# extracts content from ctv articles
async def ctv_extractor(url, person, event_date):
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        print(f"[DEBUG] Navigating to {url} from CTV")
        await page.goto(url)

        metadata = await get_dynamic_metadata(page)
        publisher = "CTV"
        published_on = get_publication_date(metadata)

        headline = await page.locator("h1.b-headline").all_text_contents() or []
        # dont get image caption 
        # caption = []
        # try:
        #     caption_content = await page.locator("figure.c-media-item").first.text_content()
        #     if caption_content:
        #         caption = [caption_content]
        # except:
        #     pass        
        body = await page.locator('article.b-article-body :is(p, h1, h2, h3)').all_text_contents() or []
        paras = headline + body
        index = 0
        cleaned_paras = []
        for p in paras:
            p = clean_paragraph(p)
            if p is None:
                continue
            if p == "stop phrase":
                df.rechunk()
                await browser.close()
                write_article(cleaned_paras, person, publisher)
                return df
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
            cleaned_paras.append(p)
        df.rechunk()
        await browser.close()
        write_article(cleaned_paras, person, publisher)

    return df

# calls appropriate extractor for a given row entry 
async def extractor(row):
    person = row[0] 
    event_date = row[1]
    url = row[2] 
    df = pl.DataFrame(schema=schema)
    try:
        if url.startswith("https://www.cbc.ca"):
            df = await cbc_extractor(url, person, event_date)
        elif url.startswith("https://www.ctvnews.ca"):
            df = await ctv_extractor(url, person, event_date)
        else:
            df = await static_extractor(url, person, event_date)
    except Exception as e:
        # return on error, continue to next article.
        print(f"Error processing {url} for {person}: {str(e)}")
    return df

# processes an csv
async def process_articles():
    data = pl.read_csv("./data/csv_datasets/batch_two.csv", 
                       has_header=True, 
                       try_parse_dates=True)
    df = pl.DataFrame(schema=schema)
    
    BATCH_SIZE = 5 # process in batches to avoid rate limits
    for i in range(0, len(data), BATCH_SIZE):
        batch = data.slice(i, BATCH_SIZE)
        articles = [extractor(row) for row in batch.rows()]
        results = await asyncio.gather(*articles)
        
        for result in results:
            if not result.is_empty():
                df = df.vstack(result)      
        await asyncio.sleep(2)  # sleep for 2 seconds to avoid rate limits
    
    # writes a CSV file - not needed as JSON articles are written
    # print(f"Final dataset size: {len(df)}")
    # df.write_csv("./data/datasets/processed_dataset_2.csv") # Replace with path-to-your-CSV

asyncio.run(process_articles())


