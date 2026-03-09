import feedparser
from urllib.parse import quote
from datetime import timedelta

def fetch_extreme_move_news(ticker, start_date, end_date):
    """
    Fetches top 3 Google News headlines for a given ticker or index during the specified week.
    start_date and end_date should be datetime objects.
    """
    # Format dates to YYYY-MM-DD
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Map index tickers to common names for better search results
    search_term = ticker
    if ticker == '^NSEI':
        search_term = "NIFTY 50"
    elif ticker == '^NSEBANK':
        search_term = "BANKNIFTY"
    elif ticker == '^BSESN':
        search_term = "SENSEX"
    elif ticker.endswith('.NS'):
        search_term = ticker.replace('.NS', '')
        
    query = f'"{search_term}" after:{start_str} before:{end_str}'
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(url)
    
    if not feed.entries:
        return []
        
    top_headlines = []
    for entry in feed.entries[:3]:
        top_headlines.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published if hasattr(entry, 'published') else "Unknown"
        })
        
    return top_headlines
