from bs4 import BeautifulSoup
import json
import polars as pl
from datetime import datetime
import asyncio
import aiohttp


schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8, "URL": pl.Utf8}


async def getHTML(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

## static helpers
def getMetaData(soup):
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
# these two might need more robust error handling
def getPublisher(news_article):
    pub = news_article.get("publisher", {})
    # pub_name = pub.get("name")
    return pub.get("name")
def getPublicationDate(news_article):
    pub_date = news_article.get("datePublished")
    if pub_date:
        date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        return date.date()
    return None
# returns text with line breaks
def extractAllText(soup):
    paragraphs = soup.article.find_all('p')
    text = soup.title.get_text()
    for p in paragraphs:
        text += "\n"
        text += p.get_text() 
    return text
def addData(publisher, url, publication_date, text, person, event_date):
    lines = text.split("\n")
    df = pl.DataFrame(schema=schema)
    for i in range(len(lines)):
        temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": publication_date, "Publisher": publisher, "URL":url, "Paragraph Index": [i], "Paragraph Text": lines[i]})
        df = df.vstack(temp)
        i += 1
    df.rechunk()
    return df

async def static_extractor(url, person, event_date):
    html_content = await getHTML(url)
    soup = BeautifulSoup(html_content, features="lxml")
    metadata = getMetaData(soup)
    publisher = getPublisher(metadata)
    publication_date = getPublicationDate(metadata)
    text = extractAllText(soup)
    return addData(publisher, url, publication_date, text, person, event_date)