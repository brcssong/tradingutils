from modules.data_retrieval_module import DataRetrieval, modify_date
from modules.sentiment_module import Sentiment
from modules.model_module import Model
from config.constants import FILENAMES

import pandas as pd
import numpy as np
import torch

import pickle
import os
import time


class Pipeline:
    def __init__(self, train_date_start: str, train_date_end: str):
        self.ts = train_date_start
        self.te = modify_date(train_date_end, 1, "D")
        assert self.ts < self.te, "The start date must be less than the end date."

        np.random.seed(100)
        torch.manual_seed(100)

    def check_cache(self, filename, pand=False):
        filename = f"cache/{filename}"
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                obj = pickle.load(f)
                if pand:
                    return obj.values.any()
                else:
                    return obj

    def write_to_cache(self, filename, obj):
        filename = f"cache/{filename}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.isfile(filename):
            print(f"WARNING: file already exists at {filename}. Overwriting.")
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    # TODO: change age to accurate age from desired date of execution, not current-day.
    def execute(self):
        print("Kicking off! ")
        print("Step 1: Getting tickers.")
        if self.check_cache(FILENAMES[0]):
            tickers, ages = self.check_cache(FILENAMES[0])
        else:
            tickers, ages = list(map(list, zip(*DataRetrieval.get_tickers())))
            self.write_to_cache(FILENAMES[0], (tickers, ages))
            print("Sleeping for a minute to skirt rate limits...")
            time.sleep(60)

        print("Tickers are", tickers)

        print(f"Step 2: Now getting data, from {self.ts} to {self.te}")
        if self.check_cache(FILENAMES[1], True) and self.check_cache(FILENAMES[2]) and self.check_cache(FILENAMES[3], True):
            df = pd.read_pickle(f"cache/{FILENAMES[1]}")
            norm_vars = self.check_cache(FILENAMES[2])
            df_valid = pd.read_pickle(f"cache/{FILENAMES[3]}")
        else:
            df = pd.DataFrame(columns=["ticker_and_date", "ftr_year_completion_percentage", "ftr_curr_price", "ftr_past_day_ret", "ftr_past_week_ret", "ftr_past_month_ret", "ftr_past_year_ret", "ftr_stock_age_days", "ftr_rsi_14", "ftr_vol_14", "ftr_recency", "ftr_past_week_market_sentiment", "ftr_curr_market_sentiment", "lbl_next_three_days_ret", "lbl_next_week_ret", "lbl_next_two_weeks_ret", "lbl_next_month_ret", "lbl_next_two_months_ret"])
       
            sentiment = Sentiment()
            while self.ts != self.te:
                old = self.ts
                print(f"Adding values from date {old}...")
                df, self.ts = DataRetrieval.get_rows_by_date(df, tickers, ages, self.ts, sentiment)
                print(f"Added values from date {old}. DataFrame is now of length {df.shape[0]}.")
                os.makedirs(os.path.dirname(f'csvs/after-{old}.csv'), exist_ok=True)
                df.to_csv(f'csvs/after-{old}.csv')
        
            print(f"Ingested data into DataFrame. Now performing full normalization.")
            df, norm_vars = DataRetrieval.full_normalization(df)
            os.makedirs(os.path.dirname(f'results/final_df.csv'), exist_ok=True)
            df.to_csv(f'results/final_df.csv')
            df.to_pickle(f"cache/{FILENAMES[1]}")

            # NOTE: validation is collected from a 15-day period
            validation_date_start, validation_date_end = modify_date(self.te, -300, "D"), modify_date(self.te, -285, "D")
            df_valid = pd.DataFrame(columns=["ticker_and_date", "ftr_year_completion_percentage", "ftr_curr_price", "ftr_past_day_ret", "ftr_past_week_ret", "ftr_past_month_ret", "ftr_past_year_ret", "ftr_stock_age_days", "ftr_rsi_14", "ftr_vol_14", "ftr_recency", "ftr_past_week_market_sentiment", "ftr_curr_market_sentiment", "lbl_next_three_days_ret", "lbl_next_week_ret", "lbl_next_two_weeks_ret", "lbl_next_month_ret", "lbl_next_two_months_ret"])

            while validation_date_start != validation_date_end:
                print(f"Adding validation values from date {validation_date_start}...")
                df_valid, validation_date_start = DataRetrieval.get_rows_by_date(df_valid, tickers, ages, validation_date_start, sentiment)
            
            df_valid = DataRetrieval.transform(df_valid, norm_vars)
            df_valid.to_pickle(f"cache/{FILENAMES[3]}")

            with open('results/norm_vars.pkl', 'wb') as file:
                pickle.dump(norm_vars, file)
            self.write_to_cache(FILENAMES[2], norm_vars)

        print(f"Step 3: Now training model!")
        model = Model()
        model.train(df, df_valid)