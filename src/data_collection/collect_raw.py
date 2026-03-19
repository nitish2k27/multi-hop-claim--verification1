import time
import pandas as pd
import feedparser
from newspaper import Article

print("Collecting news articles...")

sources = [
    {"url": "https://feeds.reuters.com/reuters/topNews",
     "source": "reuters.com", "language": "en", "category": "world"},
    {"url": "https://feeds.bbci.co.uk/news/rss.xml",
     "source": "bbc.co.uk", "language": "en", "category": "world"},
    {"url": "https://www.theguardian.com/world/rss",
     "source": "guardian.com", "language": "en", "category": "world"},
    {"url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
     "source": "timesofindia.com", "language": "en", "category": "india"},
    {"url": "https://indianexpress.com/feed/",
     "source": "indianexpress.com", "language": "en", "category": "india"},
    {"url": "https://www.snopes.com/feed/",
     "source": "snopes.com", "language": "en", "category": "fact-check"},
    {"url": "https://www.factcheck.org/feed/",
     "source": "factcheck.org", "language": "en", "category": "fact-check"},
]

articles = []

for src in sources:
    print(f"Scraping {src['source']}...")
    try:
        feed = feedparser.parse(src["url"])
        for entry in feed.entries[:200]:
            try:
                article = Article(entry.link)
                article.download()
                article.parse()
                if article.text and len(article.text) > 200:
                    articles.append({
                        "url": entry.link,
                        "title": article.title or "",
                        "text": article.text[:3000],
                        "source": src["source"],
                        "publish_date": str(article.publish_date or ""),
                        "language": src["language"],
                        "category": src["category"]
                    })
                time.sleep(0.3)
            except:
                continue
        print(f"  Total so far: {len(articles)}")
    except Exception as e:
        print(f"  Skipped {src['source']}: {e}")
        continue

# AUTO SAVES to data/raw/
pd.DataFrame(articles).to_csv(
    "data/raw/news_articles.csv", 
    index=False
)
print(f"DONE → data/raw/news_articles.csv ({len(articles)} articles)")