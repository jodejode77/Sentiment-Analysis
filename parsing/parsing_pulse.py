import requests
import pandas as pd
from datetime import datetime
import os

HEADERS = {
    'user-agent': 'Mozilla/5.0',
    'accept': '*/*'
}

# Function to fetch news from Tinkoff API
def fetch_news(ticker, cursor=''):
    url = f'https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}?cursor={cursor}&limit=50'
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Invalid JSON response for ticker {ticker}. Skipping...")
            return [], None

        data = json_data.get('payload', {}).get('items', [])
        next_cursor = json_data.get('payload', {}).get('nextCursor', None)

        news_list = []
        for item in data:
            post_date = datetime.strptime(item.get('inserted'), '%Y-%m-%dT%H:%M:%S.%f%z')
            if 2020 <= post_date.year <= 2025:
                # Count reactions of interest
                reactions = item.get('reactions', {}).get('counters', [])
                reaction_sum = sum(
                    reaction.get('count', 0)
                    for reaction in reactions
                    if reaction.get('type') in ['like', 'rocket', 'buy-up']
                )

                news = {
                    'ticker': ticker,  # Add the ticker symbol
                    'post_id': item.get('id'),
                    'owner_id': item.get('owner', {}).get('id'),
                    'date': post_date.strftime('%Y-%m-%d'),
                    'reactions_sum': reaction_sum,
                    'text': item.get('content', {}).get('text', '')
                }
                news_list.append(news)
        return news_list, next_cursor

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {ticker}: {e}")
        return [], None  # Skip batch if there's a network issue

def save_to_csv(news_data, filename='news_data_2_project.csv'):
    df = pd.DataFrame(news_data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def collect_data():
    tickers = ['SBER', 'MOEX', 'GMKN', 'MTSS', 'AFLT']

    for ticker in tickers:
        cursor = ''
        while True:
            news_batch, cursor = fetch_news(ticker, cursor)
            if not news_batch or cursor is None:
                break
            save_to_csv(news_batch)

if __name__ == '__main__':
    collect_data()
