import re
from datetime import datetime, timedelta
from typing import List

import yfinance as yf
from dagster import (
    AssetExecutionContext,
    Config,
    Definitions,
    MaterializeResult,
    MetadataValue,
    RunConfig,
    asset,
    define_asset_job,
)


def yesterday() -> str:
    yesterday = datetime.now() - timedelta(1)
    return datetime.strftime(yesterday, "%Y-%m-%d")


class YahooTickerConfig(Config):
    start_date: str
    end_date: str
    symbols: List[str]


@asset
def fetch_ticker_data(
    context: AssetExecutionContext, config: YahooTickerConfig
) -> MaterializeResult:
    symbols = config.symbols
    start_date = config.start_date
    if not re.match(start_date, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"):
        start_date = "2000-01-01"
    end_date = config.end_date
    if not re.match(end_date, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"):
        end_date = yesterday()

    tickers = yf.Tickers(symbols)
    tickers_hist_df = tickers.history(start=start_date, end=end_date)

    # TRANSFORM MULTI-LEVEL INDEX INTO A SINGLE-INDEX SET OF COLUMNS.
    tickers_hist_df = (
        tickers_hist_df.stack(level=1)
        .rename_axis(["Date", "Ticker"])
        .reset_index(level=1)
    )

    context.log.info(f"DataFrame has {len(tickers_hist_df)} rows.")

    return MaterializeResult(
        metadata={
            "num_records": len(tickers_hist_df),
            "preview": MetadataValue.md(tickers_hist_df.head().to_markdown()),
        }
    )


yahoo_ticker_job = define_asset_job(
    "yahoo_ticker_job",
    selection="fetch_ticker_data",
    config=RunConfig(
        {
            "fetch_ticker_data": YahooTickerConfig(
                start_date="2000-01-01",
                end_date="2024-01-01",
                symbols=["AAPL", "GOOGL", "ORCL", "MSFT", "IBM", "AMZN"],
            )
        }
    ),
)

defs = Definitions(
    assets=[fetch_ticker_data],
    jobs=[yahoo_ticker_job],
)
