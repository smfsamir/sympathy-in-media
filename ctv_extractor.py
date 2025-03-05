from playwright.async_api import async_playwright
import polars as pl
import re
from datetime import datetime
import html
import json

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

    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Check if text matches any stop phrases
    for phrase in stop_phrases:
        if phrase in text_lower:
            print(f"STOP PHRASE DETECTED: {phrase}")
            return "STOP PHRASE"
    
    # remove lines with just one word or less
    if len(text.split()) <= 1:
        return None
    # Remove extra whitespace
    text = ' '.join(text.split())
    # remove links and copyright
    if "href" in text:
        text = text.split("<a href", 1)[0]
    if "©" in text:
        return None
    # Filter out advertisement or empty strings
    if text.lower() in ['advertisement', ''] or len(text.strip()) < 3:
        return None
    return text

# generic function to extract publication date from metadata 
def get_publication_date(metadata):
    """Extract publication date from metadata"""
    if metadata == None:
        # get the metadata from meta tags
        # <meta property="og:date_published" content="CTV"> for example
        return
    if not metadata:
        return None
        
    pub_date = metadata.get("datePublished")
    if not pub_date:    
        pub_date = metadata.get("dateModified")
    if not pub_date:
        pub_date = metadata.get("dateUpdated")
    if not pub_date:
        pub_date = metadata.get("dateCreated")
    
    if pub_date:
        try:
            date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            return date.date()
        except ValueError:
            return None
    return None

async def ctv_extractor(url, person, event_date):
    schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8}
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url)

        metadata = await get_metadata(page)
        publisher = "CTV"
        published_on = get_publication_date(metadata)

        headline = await page.locator("h1.b-headline").all_text_contents()
        caption = [await page.locator("figure.c-media-item").first.text_content()]
        body = await page.locator('article.b-article-body :is(p, h1, h2, h3)').all_text_contents()        
        paras = headline + caption + body
        index = 0
        for p in paras:
            p = clean_paragraph(p)
            if p is None:
                continue
            if p == "STOP PHRASE":
                print("BREAKING OUT STOP PHRASE DETECTED AT INDEX", index,"FOR PERSON: ", person)
                # stop_reading = True
                # break
                df.rechunk()
                await browser.close()
                return df
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
        df.rechunk()

        await browser.close()
    return df