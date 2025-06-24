import os
from dotenv import load_dotenv

load_dotenv()
print("BINANCE_API_KEY:", os.getenv("BINANCE_API_KEY") is not None)
print("BINANCE_API_SECRET:", os.getenv("BINANCE_API_SECRET") is not None)