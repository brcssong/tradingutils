from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle

from config.client_setup import session
from config.constants import FILENAMES
from modules.data_retrieval_module import DataRetrieval
from modules.sentiment_module import Sentiment
from modules.model_module import Model

class Prediction:
    def __init__(self):
        self.model_class = Model()
        self.model_class.load()
        self.sentiment = Sentiment()

    def check_cache(self, filename, pand=False):
        filename = f"cache/{filename}"
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                obj = pickle.load(f)
                if pand:
                    return obj.values.any()
                else:
                    return obj
        
    def predict(self, tickers=["AAPL", "GOOG", "NVDA", "AMZN", "META", "AVGO", "NFLX", "TSLA", "PLTR", "COIN"], date=date.today().strftime("%Y-%m-%d")):
        print("Kicking off! ")
        print("Step 1: Getting tickers.")
    
        ages_in_days = []
        print(tickers)
        for ticker in tickers:
            yf_ticker = yf.Ticker(ticker, session=session)
            history = yf_ticker.history(period="max", auto_adjust=False)
            if not history.shape[0]:
                continue
            first_date = history.index[0]
            now = pd.Timestamp.now(tz=first_date.tz)

            ages_in_days.append((now - first_date) / np.timedelta64(1, 'D'))

        print("Step 2: Collecting data and generating dataframe.")
        df = pd.DataFrame(columns=["ticker_and_date", "ftr_year_completion_percentage", "ftr_curr_price", "ftr_past_day_ret", "ftr_past_week_ret", "ftr_past_month_ret", "ftr_past_year_ret", "ftr_stock_age_days", "ftr_rsi_14", "ftr_vol_14", "ftr_recency", "ftr_past_week_market_sentiment", "ftr_curr_market_sentiment", "lbl_next_three_days_ret", "lbl_next_week_ret", "lbl_next_two_weeks_ret", "lbl_next_month_ret", "lbl_next_two_months_ret"])
        df, _ = DataRetrieval.get_rows_by_date(df, tickers, ages_in_days, date, self.sentiment)
        if not df.shape[0]:
            raise Exception("Is the day you chose a trading day or in the past/present?")
        assert self.check_cache(FILENAMES[2]), "normalization variable information does not exist on disk."

        # Normalize
        df_normed = DataRetrieval.transform(df, self.check_cache(FILENAMES[2]))

        print("df_normed is...")
        print(df_normed)

        dataloader = self.model_class.create_dataloader_for_pred(df_normed)
        preds = self.model_class.test(dataloader)

        # De-norm
        df_denormed = DataRetrieval.inverse_transform(preds, self.check_cache(FILENAMES[2]))
        df_denormed["ticker_and_date"] = df["ticker_and_date"]

        print("df_denormed is...")
        print(df_denormed)

        return df_denormed