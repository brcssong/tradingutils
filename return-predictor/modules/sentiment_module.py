from waybackpy import WaybackMachineCDXServerAPI
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np
import time

from config.constants import NEWS_SOURCES

def get_snapshot_from_url(url: str, date: str):
    wayback = WaybackMachineCDXServerAPI(url, user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")

    y, m, d = map(int, date.split("-"))

    near = wayback.near(year=y, month=m, day=d, hour=8, minute=0)
    return near.archive_url

def extract_headlines(snapshot_url: str, html_element: str):
    try:
        response = requests.get(snapshot_url, timeout=60)
        soup = BeautifulSoup(response.content, "html.parser")
        headlines = [tag.get_text(strip=True) for tag in soup.find_all(html_element)]
        return [h for h in headlines if len(h) > 45]
    except Exception as e:
        print(f"Error at {snapshot_url}: {e}")
        return []

def score(result: list[dict]) -> float:
    weights = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    score = sum(weights[r['label']] * r['score'] for r in result)

    # bias so that scores near 0 are more preferable, and near 1 and -1 are less preferable
    bias = lambda x: -0.25*(x*x) + 1
    return bias(score) * round(score, 4)

class Sentiment:
    def __init__(self):
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3).to('mps')
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, top_k=None)

    def process_scores(self, headline_results: list[dict]):
        scores = np.array([score(r) for r in headline_results])
        return np.mean(scores)

    def get_score_on_news(self, date: str) -> float:
        headlines = []
        for source_name, (source_url, html_element) in NEWS_SOURCES.items():
            print(f"Sentiment: Starting news collection for {source_name} on date {date}...")
            snapshot = get_snapshot_from_url(source_url, date)
            print(f"Sentiment: Getting headlines for this source...")
            headlines.extend(extract_headlines(snapshot, html_element))
            time.sleep(15)
        
        results = self.nlp(headlines)
        print(f"Sentiment: Finished sentiment analysis on date {date}")
        return self.process_scores(results)