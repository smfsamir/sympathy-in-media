from playwright.async_api import async_playwright
import polars as pl
import re
from datetime import datetime
import html
import json
from ctv_extractor import ctv_extractor
from cbc_extractor import cbc_extractor

async def get_metadata(page):
    """Extract metadata from JSON-LD script tags using Playwright"""
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

def clean_paragraph(text):
    stop_phrases = [
        'related articles',
        'related stories',
        'related news',
        'recommended videos',
        'more on',
        'more from',
        'most read',
        'trending now',
        'you may also like',
        'recommended for you',
        'further reading',
        'sponsored content',
        'more stories',
        'read next',
        'other news',
        'popular articles',
        'top stories',
        'top news',
        'top articles',
        'most popular',
        'trending videos',
        'trending articles',
        'trending news',
        'trending stories',
        'trending now',
    ]

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
    account_words = [
        "subscribe now",
        "create an account",
        "sign up for an account",
        "sign in if you have an account",
        "sign-in if you have an account",
        "log in if you have an account",
        "sign in to your account",
        "sign-in to your account",
        "log in to your account"
    ]
    for word in account_words:
        if word in text.lower():
            return None

    return text

async def get_publisher(page, metadata, url):
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

async def dynamic_extractor(url, person, event_date):
    schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8}
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url)

        metadata = await get_metadata(page)
        publisher = await get_publisher(page, metadata, url)
        published_on = get_publication_date(metadata)

        paras = await page.locator('article :is(p, h1, h2, figcaption)').all_text_contents()
        if not paras:
            paras = await page.locator(':is(p, h1, h2, figcaption)').all_text_contents()

        index = 0
        stop_reading = False
        for p in paras:
            if stop_reading:
                break
            p = clean_paragraph(p)
            if p is None:
                continue
            if p == "stop phrase":
                df.rechunk()
                await browser.close()
                return df
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
        df.rechunk()

        await browser.close()
    return df