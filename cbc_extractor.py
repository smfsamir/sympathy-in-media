from playwright.async_api import async_playwright
import polars as pl
import re
from datetime import datetime
import html


def dateExtractor(datestr):
    # getting just the original publication date (cbc specfic)
    published_on = datestr.split(': ')[1].split(' |')[0]
    # convert to date obj for df
    # will work for Mon or Month DD, YYYY format so pretty cbc specific but can probably change the regex and generalize this function
    date_found = re.search('([A-Za-z]{3,9} \d{1,2}, \d{4})', published_on)
    if date_found:
        date = datetime.strptime(date_found.group(1), '%b %d, %Y')
        return date.date()
    # shouldn't get here
    return None


async def cbc_extractor(url, person, event_date):
    schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8}
    df = pl.DataFrame(schema=schema)
    async with async_playwright() as p:

        browser = await p.webkit.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        print("getting page for: ", person)
        await page.goto(url)
        print("got PAGE for: ", person)


        # get/set data that will be re-used first
        publisher = "CBC"
        date_info = await page.locator('time.timeStamp').first.text_content()
        published_on = dateExtractor(date_info)


        # get headline
        headline = await page.locator("h1.detailHeadline").all_text_contents()
        print("got headline for: ", person)


        # get byline
        byline = await page.locator("h2.deck").all_text_contents()

        # get first img caption
        img = [await page.locator("figcaption").first.text_content()]  
   
        # get article body and img captions
        # todo
        # - check that strange artifacts (show up in excel) don't show up when we pull text from df
        body = await page.locator(".story p:not(article p), .story figcaption:not(article figcaption), .story h2:not(article h2)").all_text_contents()

        paras = headline + byline + img + body

        # adding to the df
        index = 0
        for p in paras:
            if "href" in p:
                p = p.split("<a href", 1)[0]
            if p == "":
                continue
            temp = pl.DataFrame({"Person Name": person, "Incident Date": event_date, "Publication Date": published_on, "Publisher": publisher, "URL":url, "Paragraph Index": [index], "Paragraph Text": p})
            df = df.vstack(temp)
            index += 1
        df.rechunk()

        await browser.close()
    return df