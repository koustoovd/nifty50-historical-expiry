import feedparser
from urllib.parse import quote
from datetime import datetime

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 10)
query = f'"SENSEX" after:2023-01-01 before:2023-01-10'
encoded_query = quote(query)
url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
print(url)
feed = feedparser.parse(url)
print("Entries:", len(feed.entries))
if feed.entries:
    print(feed.entries[0].title)
