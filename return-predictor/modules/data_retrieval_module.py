from config.client_setup import api, session
from config.constants import MIN_ALLOWED_CURR_PRICE
from modules.sentiment_module import Sentiment

from alpaca.trading.enums import AssetExchange, AssetStatus

import yfinance as yf
import talib
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Union
import time
import pandas_market_calendars as mcal

def pre_get_filtered_assets() -> list[str]:
    is_desired_asset = lambda asset: asset.tradable and asset.easy_to_borrow and asset.exchange == AssetExchange.NASDAQ and asset.status == AssetStatus.ACTIVE
    return [asset.symbol for asset in api.get_all_assets() if is_desired_asset(asset)]

def is_index_fund(info):
    if info.get("quoteType") not in ["ETF", "MUTUALFUND"]:
        return False
    name = info.get("longName", "").lower()
    return any(keyword in name for keyword in ["index", "s&p", "500", "total market"])

def is_stable_stock(data, window_days=30, threshold=5):
    try:
        if data.empty or len(data) < window_days:
            return False

        prices = data['Adj Close']

        rolling_max = prices.rolling(window=window_days).max()
        rolling_min = prices.rolling(window=window_days).min()

        ratio = (rolling_max / rolling_min.replace(0, pd.NA)).dropna()

        if (ratio >= threshold).any():
            return False
        return True
    except:
        print(f"Error checking if stock is stable.")
        return False

def does_asset_pass(asset: str) -> tuple[bool, float]:
    # Returns market cap
    ticker = yf.Ticker(asset, session=session)
    age_in_days = -1
    try:
        if not ticker:
            raise Exception()

        # Exclude index funds
        if is_index_fund(ticker.info):
            raise Exception()
        
        # Exclude non-USA funds
        if ticker.info.get("country", "") != "United States":
            raise Exception()

        # Exclude stocks younger than 5 year
        history = ticker.history(period="max", auto_adjust=False)
        first_date = history.index[0]
        now = pd.Timestamp.now(tz=first_date.tz)

        age_in_days = (now - first_date) / np.timedelta64(1, 'D')
        if age_in_days < 5 * 365:
            raise Exception()

        # Exclude stocks with less than $8 current value
        last_close = history['Adj Close'].iloc[-1]
        if last_close < MIN_ALLOWED_CURR_PRICE:
            raise Exception()
        
        # Exclude stocks that have fluctuated more than 5x up or down in a single month
        if not is_stable_stock(history):
            raise Exception()
    except:
        return [False, -1]
    return [True, age_in_days]

def post_get_filtered_assets(assets: list[str]) -> tuple[bool, float]:
    # returns whether ticker is valid to add to list or not
    validated_tickers = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(does_asset_pass, assets))
        for asset, res in zip(assets, results):
            should_add, age_in_days = res
            should_add and validated_tickers.append((asset, age_in_days))
    return validated_tickers

def get_ticker_stats(assetDate: tuple[pd.DataFrame, str]) -> tuple[Union[int, float]]:
    asset_history, date = assetDate

    end_date = date
    start_date = modify_date(date, -45, "D")

    history = asset_history.loc[(asset_history.index > start_date) & (asset_history.index <= end_date)]
    history = history.tail(15)
    
    # Get RSI-14
    rsi = talib.RSI(history.astype('float64').values, timeperiod=14)[-1]
    # Get volatility-14
    history = history.tail(14)
    returns = history.pct_change()
    volatility = returns.std()

    return rsi, volatility

def process_get_ticker_stats(asset_histories: list[pd.DataFrame], date: str) -> list[tuple[Union[int, float]]]:
    # returns data points of tickers [mkt_cap, rsi-14, volatility-14]
    asset_comb_obj = zip(asset_histories, [date] * len(asset_histories))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(get_ticker_stats, asset_comb_obj))
        return results
        
def modify_date(date: str, amount: int, dateType: str) -> tuple[str, str]:
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    if amount == 0:
        return date
    elif amount > 0:
        match dateType:
            case "Y":
                return (date_obj + relativedelta(years=amount)).strftime('%Y-%m-%d')
            case "M":
                return (date_obj + relativedelta(months=amount)).strftime('%Y-%m-%d')
            case "W":
                return (date_obj + relativedelta(weeks=amount)).strftime('%Y-%m-%d')
            case "D":
                return (date_obj + relativedelta(days=amount)).strftime('%Y-%m-%d')
            case _:
                return date
    else:
        match dateType:
            case "Y":
                return (date_obj - relativedelta(years=abs(amount))).strftime('%Y-%m-%d')
            case "M":
                return (date_obj - relativedelta(months=abs(amount))).strftime('%Y-%m-%d')
            case "W":
                return (date_obj - relativedelta(weeks=abs(amount))).strftime('%Y-%m-%d')
            case "D":
                return (date_obj - relativedelta(days=abs(amount))).strftime('%Y-%m-%d')
            case _:
                return date
    return date

def weight_decay(t: int) -> float:
    # half-life of 14 days (~10 trading days)
    lmb = -np.log(2) / 14
    return np.exp(lmb * t)

def date_nearest(date: str, id: pd.Index) -> str:
    if date not in id:
        nearest_idx = id.get_indexer([date], method='nearest')[0]
        return id[nearest_idx].strftime('%Y-%m-%d')
    return date

def percentage_of_year_compl(date: str) -> float:
    _, month, day = date.split("-")
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30] # Assuming year is not a leap
    total_days = 0
    for m in range(int(month) - 1):
        total_days += days[m]
    total_days += int(day)
    return total_days / 365

def get_price(histories: pd.DataFrame, date: str, ticker_code: str) -> float:
    return histories.loc[date][ticker_code]

def calc_rel_normalized_return(histories: pd.DataFrame, first_date: str, new_date: str, ticker_code: str) -> float:
    ticker_rel_return = np.log(get_price(histories, new_date, ticker_code) / get_price(histories, first_date, ticker_code))
    idxf_rel_return = np.log(get_price(histories, new_date, "SPY") / get_price(histories, first_date, "SPY"))
    return ticker_rel_return - idxf_rel_return

def market_is_open(date):
    result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    return result.empty == False

class DataRetrieval:
    def get_tickers() -> list[tuple[bool, float]]:
        # Pick random sample of 75 stocks
        stocks = post_get_filtered_assets(pre_get_filtered_assets())
        np.random.seed(100)
        chosen_indices = np.random.choice(a=len(stocks), size=min(75, len(stocks)), replace=False)
        result = [stocks[i] for i in chosen_indices]
        return result

    def get_rows_by_date(df: pd.DataFrame, tickers: list[str], ages_in_days: list[float], date: str, sentiment: Sentiment, include_targets=True) -> tuple[pd.DataFrame, str]:
        # Retrieves data for a given timeframe on all tickers
        # This will be -2 yrs from date -> date -> +2 months from date

        if not market_is_open(date):
            return df, modify_date(date, 1, "D")
        time.sleep(10)

        ticker_codes = tickers[:]

        tickers = yf.Tickers(tickers + ["SPY"], session=session)
        two_years_ago = modify_date(date, -2, "Y")
        two_months_ahead = modify_date(date, 2, "M")

        try:
            histories = tickers.history(start=two_years_ago, end=two_months_ahead, session=session, auto_adjust=False)["Adj Close"]
            if histories.empty:
                raise ValueError("No data retrieved")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return df, modify_date(date, 1, "D")
        
        two_years_ago = date_nearest(two_years_ago, histories.index)
        one_year_ago = date_nearest(modify_date(date, -1, "Y"), histories.index)
        one_month_ago = date_nearest(modify_date(date, -1, "M"), histories.index)
        one_week_ago = date_nearest(modify_date(date, -1, "W"), histories.index)
        one_day_ago = date_nearest(modify_date(date, -1, "D"), histories.index)  
        current = date_nearest(date, histories.index)
        if current != date:
            return df, modify_date(date, 1, "D")
        if include_targets:
            three_days_ahead = date_nearest(modify_date(date, 3, "D"), histories.index)
            one_week_ahead = date_nearest(modify_date(date, 1, "W"), histories.index)
            two_weeks_ahead = date_nearest(modify_date(date, 2, "W"), histories.index)
            one_month_ahead = date_nearest(modify_date(date, 1, "M"), histories.index)
            two_months_ahead = date_nearest(two_months_ahead, histories.index)

        current_sentiment = sentiment.get_score_on_news(date)
        time.sleep(60)
        week_ago_sentiment = sentiment.get_score_on_news(one_week_ago)

        histories_arr = [histories[t] for t in ticker_codes]

        ticker_stats = process_get_ticker_stats(histories_arr, date)

        # placing features into row and adding row to DataFrame
        rows = []

        for idx, code in enumerate(tqdm(ticker_codes)):
            rsi_14, volatility_14 = ticker_stats[idx]
            stock_age = ages_in_days[idx]
            if include_targets:
                df_row = {
                    "ticker_and_date": f"{code}&{date}",
                    "ftr_year_completion_percentage": percentage_of_year_compl(current),
                    "ftr_curr_price": get_price(histories, current, code),
                    "ftr_past_day_ret": calc_rel_normalized_return(histories, one_day_ago, current, code),
                    "ftr_past_week_ret": calc_rel_normalized_return(histories, one_week_ago, current, code),
                    "ftr_past_month_ret": calc_rel_normalized_return(histories, one_month_ago, current, code),
                    "ftr_past_year_ret": calc_rel_normalized_return(histories, one_year_ago, current, code),
                    "ftr_stock_age_days": stock_age,
                    "ftr_rsi_14": rsi_14,
                    "ftr_vol_14": volatility_14,
                    "ftr_recency": 1,
                    "ftr_past_week_market_sentiment": week_ago_sentiment,
                    "ftr_curr_market_sentiment": current_sentiment,
                    "lbl_next_three_days_ret": calc_rel_normalized_return(histories, current, three_days_ahead, code),
                    "lbl_next_week_ret": calc_rel_normalized_return(histories, current, one_week_ahead, code),
                    "lbl_next_two_weeks_ret": calc_rel_normalized_return(histories, current, two_weeks_ahead, code),
                    "lbl_next_month_ret": calc_rel_normalized_return(histories, current, one_month_ahead, code),
                    "lbl_next_two_months_ret": calc_rel_normalized_return(histories, current, two_months_ahead, code)
                }
            else:
                df_row = {
                    "ticker_and_date": f"{code}&{date}",
                    "ftr_year_completion_percentage": percentage_of_year_compl(current),
                    "ftr_curr_price": get_price(histories, current, code),
                    "ftr_past_day_ret": calc_rel_normalized_return(histories, one_day_ago, current, code),
                    "ftr_past_week_ret": calc_rel_normalized_return(histories, one_week_ago, current, code),
                    "ftr_past_month_ret": calc_rel_normalized_return(histories, one_month_ago, current, code),
                    "ftr_past_year_ret": calc_rel_normalized_return(histories, one_year_ago, current, code),
                    "ftr_stock_age_days": stock_age,
                    "ftr_rsi_14": rsi_14,
                    "ftr_vol_14": volatility_14,
                    "ftr_recency": 1,
                    "ftr_past_week_market_sentiment": week_ago_sentiment,
                    "ftr_curr_market_sentiment": current_sentiment
                }
            rows.append(df_row)
        
        df = pd.concat([df, pd.DataFrame(rows)])

        return df, modify_date(date, 1, "D")

    def augment_tickered_row(df: pd.DataFrame, index: str):
        pass # TODO for better validation accuracy

    def transform(df: pd.DataFrame, norm_vars: dict) -> pd.DataFrame:
        df = df.copy()
        for col, (mean, std) in norm_vars.items():
            if std == 0:
                df[col] = 0
            else:
                df[col] = (df[col] - mean) / std
        return df

    # denormalize the data
    def inverse_transform(preds: torch.Tensor, norm_vars: dict, label_columns=["lbl_next_three_days_ret", "lbl_next_week_ret", "lbl_next_two_weeks_ret", "lbl_next_month_ret", "lbl_next_two_months_ret"]) -> pd.DataFrame:
        preds_np = preds.numpy()
        preds_denorm = []

        for i, col in enumerate(label_columns):
            mean, std = norm_vars[col]
            if std == 0:
                preds_denorm.append(np.full(preds_np.shape[0], mean))
            else:
                preds_denorm.append(preds_np[:, i] * std + mean)

        df_denorm = pd.DataFrame(np.stack(preds_denorm, axis=1), columns=label_columns)
        return df_denorm


    # normalize the data
    def full_normalization(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        norm_vars = dict()

        cols = [c for c in df.columns if c != "ticker_and_date" and c != "ftr_year_completion_percentage"]

        for col in cols:
            mean = df[col].mean()
            std = df[col].std()

            if std == 0:
                print(f"Warning: Standard deviation is 0 for column '{col}'. Skipping normalization.")
                df[col] = 0
                norm_vars[col] = (mean, std)
                continue
            
            df[col] = (df[col] - mean) / std
            norm_vars[col] = (mean, std)

        return df, norm_vars
