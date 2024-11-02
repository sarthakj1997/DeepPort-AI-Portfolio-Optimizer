# src/sentiment_analysis.py

from src.database import get_engine, create_tables, get_session, SentimentData
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime

class SentimentAnalysis:
    def __init__(self, tickers):
        self.tickers = tickers
        self.engine = get_engine()
        create_tables(self.engine)
        self.session = get_session(self.engine)
    
    def get_news_headlines(self, ticker):
        
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(req.content, 'html.parser')
        news_table = soup.find(id='news-table')
        headlines = []
        if news_table:
            for row in news_table.findAll('tr'):
                text = row.a.get_text()
                date_data = row.td.text.strip()
                #date_data = date_data.replace('Today', datetime.now().strftime('%b-%d-%y'))
                # date_data = datetime.strptime(date_data, '%b-%d-%y %I:%M%p')
                if len(date_data) > 8:
                    
                    date_data = date_data.replace('Today', datetime.datetime.now().strftime('%b-%d-%y'))
                    date = datetime.datetime.strptime(date_data, '%b-%d-%y %I:%M%p')
                    
                else:
                    
                    date = datetime.datetime.strptime(date_data, '%I:%M%p')
                    date = datetime.datetime.combine(datetime.datetime.now().date(), date.time())
                headlines.append((date, text))
        return headlines
    
    def analyze_and_store_sentiment(self):
        for ticker in self.tickers:
            headlines = self.get_news_headlines(ticker)
            for date, headline in headlines:
                analysis = TextBlob(headline)
                sentiment_score = analysis.sentiment.polarity
                sentiment_data = SentimentData(
                    ticker=ticker,
                    date=date,
                    sentiment_score=sentiment_score
                )
                self.session.add(sentiment_data)
            self.session.commit()
