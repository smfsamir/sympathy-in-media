import asyncio
import polars as pl
from datetime import datetime
from cbc_extractor import cbc_extractor

async def extractor(row):
    person = row[0] 
    event_date = row[1]
    url = row[2] 
    if url.startswith("https://www.cbc.ca"):
        df = await cbc_extractor(url, person, event_date)
    return df


async def process_articles():
    data = pl.read_csv("./data/mini_cbc.csv", has_header=True, try_parse_dates=True)
    schema = {"Person Name": pl.Utf8, "Event Date": pl.Date, "Publication Date": pl.Date, "Publisher": pl.Utf8, "URL": pl.Utf8, "Paragraph Index": pl.Int64, "Paragraph Text": pl.Utf8, "URL": pl.Utf8}
    df = pl.DataFrame(schema=schema)
    articles = []
    for row in data.rows():
        articles.append(extractor(row))
    results = await asyncio.gather(*articles) # wait for all articles to return their dfs
    for result in results:
        df = df.vstack(result)
    print(df)
    out = "./processed_mini_cbc.csv"
    df.write_csv(out)

asyncio.run(process_articles())
