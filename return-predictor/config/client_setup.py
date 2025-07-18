import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

assert ALPACA_API_KEY is not None, "Missing ALPACA_API_KEY in .env"
assert ALPACA_SECRET_KEY is not None, "Missing ALPACA_SECRET_KEY in .env"

# Alpaca setup

from alpaca.trading.client import TradingClient
api = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=False)

# Curl_cffi setup

from curl_cffi import requests
session = requests.Session(impersonate="chrome")