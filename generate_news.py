"""
generate_news.py

Fetch headlines from your chosen RSS feeds and write docs/data/news.json.

Now also adds sentiment using NLTK VADER on the headline text.

Feeds:
- https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI
- https://www.zeebiz.com/market.xml
- https://www.financialexpress.com/market/feed/
- https://www.thehindubusinessline.com/markets/feeder/default.rss
- https://www.cnbctv18.com/news/rss/markets.xml
- https://www.livemint.com/rss/markets
"""

from pathlib import Path
from datetime import datetime
import json
import feedparser

# --- NLTK VADER setup ---
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is available (first run will download it)
try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
# Recreate after download
sia = SentimentIntensityAnalyzer()

FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI",
    "https://www.zeebiz.com/market.xml",
    "https://www.financialexpress.com/market/feed/",
    "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "https://www.cnbctv18.com/news/rss/markets.xml",
    "https://www.livemint.com/rss/markets",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ADANIENT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ADANIPORTS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMBUJACEM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ASIANPAINT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AXISBANK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BAJAJ-AUTO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BAJAJFINSV.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BAJFINANCE.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BANDHANBNK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BANKBARODA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BATAINDIA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BERGEPAINT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BHARTIARTL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BIOCON.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BOSCHLTD.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BPCL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BRITANNIA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CANBK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CHOLAFIN.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CIPLA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=COALINDIA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=COFORGE.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=COLPAL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DABUR.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DIVISLAB.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DLF.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DRREDDY.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EICHERMOT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GAIL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLAND.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GODREJCP.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GRASIM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HAVELLS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HCLTECH.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCAMC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCBANK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCLIFE.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HEROMOTOCO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HINDALCO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HINDUNILVR.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ICICIBANK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ICICIGI.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ICICIPRULI.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IDEA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IDFCFIRSTB.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INDHOTEL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INDIAMART.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INDIGO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INDUSINDBK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=INFY.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IOC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IRCTC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ITC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JINDALSTEL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JSWSTEEL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=KOTAKBANK.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=LT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=LTIM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=LUPIN.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=M&M.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MARICO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MARUTI.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MCDOWELL-N.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MPHASIS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MRF.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MUTHOOTFIN.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NATIONALUM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NESTLEIND.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NMDC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NTPC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ONGC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PAGEIND.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PEL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PFC.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PIDILITIND.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PIIND.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=PNB.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=POLYCAB.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=POWERGRID.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RECLTD.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SBI.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SBILIFE.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SHREECEM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SIEMENS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SRF.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SUNPHARMA.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATACOMM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATACONSUM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATAMOTORS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATAPOWER.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATASTEEL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TCS.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TECHM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TITAN.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TORNTPHARM.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TORNTPOWER.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TRENT.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TVSMOTOR.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ULTRACEMCO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=UPL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=VEDL.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=WIPRO.NS",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ZOMATO.NS",
]


def score_headline(text: str | None) -> dict:
    """
    Use VADER to compute sentiment for a headline.
    Returns:
      {
        "sentiment_score": float,   # compound score [-1, 1]
        "sentiment_label": str      # "positive" / "neutral" / "negative"
      }
    """
    if not text:
        return {"sentiment_score": 0.0, "sentiment_label": "neutral"}

    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"sentiment_score": compound, "sentiment_label": label}


def fetch_all_feeds(max_items_per_feed: int = 20):
    items = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title") or url
        for entry in feed.entries[:max_items_per_feed]:
            title = entry.get("title")
            sentiment = score_headline(title)
            items.append(
                {
                    "title": title,
                    "link": entry.get("link"),
                    "source": source,
                    "published": entry.get("published", ""),
                    "sentiment_score": sentiment["sentiment_score"],
                    "sentiment_label": sentiment["sentiment_label"],
                }
            )
    return items


def main():
    data_dir = Path("docs") / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    items = fetch_all_feeds()
    news_obj = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items": items,
    }
    out_path = data_dir / "news.json"
    out_path.write_text(json.dumps(news_obj, indent=2), encoding="utf-8")
    print(f"Wrote news -> {out_path}")


if __name__ == "__main__":
    main()
