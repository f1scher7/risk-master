import random
import yfinance as yf


def fetch_crypto_prices(exchange):
    try:
        btc_price = exchange.fetch_ticker('BTC/USDT')
        eth_price = exchange.fetch_ticker('ETH/USDT')
        doge_price = exchange.fetch_ticker('DOGE/USDT')

        return {
            "Bitcoin": btc_price['last'],
            "Ethereum": eth_price['last'],
            "Dogecoin": doge_price['last'],
            "PolskaGuromCoin": max(round(btc_price['last'] * random.uniform(0.7, 1.5), 2), 7)
        }
    except Exception as e:
        print(f"Error fetching crypto prices: {e}")
        return


def fetch_metals_prices():
    try:
        gold = yf.Ticker("GC=F")
        silver = yf.Ticker("SI=F")

        gold_price = gold.history(period="1d")['Close'].iloc[-1]
        silver_price = silver.history(period="1d")['Close'].iloc[-1]

        return {
            "Gold": round(float(gold_price), 2),
            "Silver": round(float(silver_price), 2)
        }
    except Exception as e:
        print(f"Error fetching metals prices: {e}")
        return


def fetch_all_prices(exchange):
    try:
        crypto_prices = fetch_crypto_prices(exchange)
        metals_prices = fetch_metals_prices()

        return {**crypto_prices, **metals_prices}
    except Exception as e:
        print(f"Error fetching all prices: {e}")
        return