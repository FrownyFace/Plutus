from __future__ import division

import pandas as pd
from sklearn.utils import shuffle
import datetime

def load_data(path):
    z = pd.read_csv(path)
    z = z[(z.option_type == "C") & (z.trade_condition_id == 0) & (z.canceled_trade_condition_id == 0) &
          (z.underlying_bid != 0) & (z.underlying_ask != 0)]

    z["expiration"] = pd.to_datetime(z["expiration"], yearfirst=True, infer_datetime_format=True)
    z["quote_datetime"] = pd.to_datetime(z["quote_datetime"], infer_datetime_format=True)
    total_seconds_year = datetime.timedelta(days=365).total_seconds()
    z["exp_date"] = z["expiration"].sub(z["quote_datetime"]).apply(lambda x: datetime.timedelta.total_seconds(x) /
                                                                             total_seconds_year)

    z = z.drop(["underlying_symbol", "quote_datetime", "sequence_number", "root", "expiration", "option_type",
                "exchange_id", "trade_condition_id", "canceled_trade_condition_id", "number_of_exchanges",
                "{exchange", "bid", "ask}[number_of_exchanges]"], axis=1)

    z["bid_size"] = z["bid_size"].apply(
        lambda x: (x - z["bid_size"].min()) / (z["bid_size"].max() - z["bid_size"].min()))
    z["ask_size"] = z["ask_size"].apply(
        lambda x: (x - z["ask_size"].min()) / (z["ask_size"].max() - z["ask_size"].min()))
    z["trade_size"] = z["trade_size"].apply(
        lambda x: (x - z["trade_size"].min()) / (z["trade_size"].max() - z["trade_size"].min()))

    underlying_min = z[["strike", "underlying_bid", "underlying_ask"]].min(axis=0).min()
    underlying_max = z[["strike", "underlying_bid", "underlying_ask"]].max(axis=0).max()
    opt_min = z[["trade_price", "best_bid", "best_ask"]].min(axis=0).min()
    opt_max = z[["trade_price", "best_bid", "best_ask"]].max(axis=0).max()

    z["strike"] = z["strike"].apply(lambda x: (x - underlying_min) / (underlying_max - underlying_min))
    z["underlying_bid"] = z["underlying_bid"].apply(lambda x: (x - underlying_min) / (underlying_max - underlying_min))
    z["underlying_ask"] = z["underlying_ask"].apply(lambda x: (x - underlying_min) / (underlying_max - underlying_min))
    z["trade_price"] = z["trade_price"].apply(lambda x: (x - opt_min) / (opt_max - opt_min))
    z["best_bid"] = z["best_bid"].apply(lambda x: (x - opt_min) / (opt_max - opt_min))
    z["best_ask"] = z["best_ask"].apply(lambda x: (x - opt_min) / (opt_max - opt_min))

    z = shuffle(z)

    y = z["trade_price"].as_matrix()
    z = z.drop("trade_price", axis=1)
    x = z.as_matrix()
    return x, y


#load_data("./data/UnderlyingOptionsTrades_2016-06-01.csv")
