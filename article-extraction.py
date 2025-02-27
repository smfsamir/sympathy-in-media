import asyncio
import polars as pl
from datetime import datetime
from playwright.async_api import async_playwright
import re
import html
from cbc_extractor import cbc_extractor
from static_extractor import static_extractor
from concurrent.futures import ThreadPoolExecutor
import polars as pl
import csv

schema = {"Person Name": pl.Utf8, "Incident Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8, "URL": pl.Utf8}

# todo
# - ctv
# - implement a bs timeout for playwright fallback
# - implement timeout + retry for playwright
# - track errors in a seperate df
async def extractor(row):
    person = row[0] 
    event_date = row[1]
    url = row[2] 
    df = pl.DataFrame(schema=schema)
    try:
        df = await static_extractor(url, person, event_date)
        # # only handles cbc and global rn, need to implement ctv
        # if url.startswith("https://www.cbc.ca"):
        #     # print("processing: ", person)
        #     df = await cbc_extractor(url, person, event_date)
        # # elif url.startswith("https://globalnews.ca"):
        # #     df = await static_extractor(url, person, event_date)
    except Exception as e:
        # return on error, continue to next article.
        print(f"Error processing {url} for {person}: {str(e)}")
    return df


async def process_articles():
    data = pl.read_csv("./data/canadian_deadly_force_cbc.csv", 
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
        # print(f"Processed batch {i//BATCH_SIZE + 1}, total rows so far: {len(df)}")
        await asyncio.sleep(2)  # added delay to avoid rate limits
    
    print(f"Final dataset size: {len(df)}")
    df.write_csv("./processed_mini_cbc.csv")

asyncio.run(process_articles())


