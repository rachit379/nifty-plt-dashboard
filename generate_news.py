"""
generate_news.py

Fetch headlines from your chosen RSS feeds and write docs/data/news.json.

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

FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NSEI",
    "https://www.zeebiz.com/market.xml",
    "https://www.financialexpress.com/market/feed/",
    "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "https://www.cnbctv18.com/news/rss/markets.xml",
    "https://www.livemint.com/rss/markets",
]


def fetch_all_feeds(max_items_per_feed: int = 20):
    items = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        source = feed.feed.get("title") or url
        for entry in feed.entries[:max_items_per_feed]:
            items.append(
                {
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "source": source,
                    "published": entry.get("published", ""),
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
